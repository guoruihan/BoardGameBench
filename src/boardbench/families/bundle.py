"""Portable frozen opponent pools. Export explicit dev selections, never rank them."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from . import evaluate as ev
from .play import make_policy, write_json
from .registry import game_class

FORMAT = 'boardbench_bot_bundle_v1'
POLICY_FIELDS = {'kind', 'game', 'config', 'rules_version', 'weights', 'path',
                 'sha256', 'stochastic', 'decision_guard'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')


def _signed(value):
    return dict(value, manifest_sha256=ev.canonical_hash(value))


def _verify_signature(value):
    unsigned = {k: v for k, v in value.items() if k != 'manifest_sha256'}
    if value.get('manifest_sha256') != ev.canonical_hash(unsigned):
        raise ValueError('Manifest checksum mismatch')


def _identity(spec):
    identity = deepcopy(spec)
    if identity['kind'] in ('neural', 'harmonies_rl'):
        identity['path'] = 'sha256:'+identity['sha256']
    return ev.canonical_hash(identity)


def _check_guard(spec):
    if 'decision_guard' in spec and (spec.get('game') != 'targi' or
            spec['decision_guard'] != 'targi_trader_acyclic_v1'):
        raise ValueError('Unknown decision_guard or wrong game')


def _member(root, relative, directory=None):
    """A package member cannot escape the package or traverse a symlink."""
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Bundle paths must be relative')
    candidate = (directory or root)/relative
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Bundle member escapes its directory or is not a file')
    # Reject symlinks even when they happen to resolve inside the bundle.
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Bundle members must not be symlinks')
    return path


def _read_dev(path, game):
    content = Path(path).read_bytes()
    manifest = json.loads(content)
    _verify_signature(manifest)
    cls = game_class(game)
    if manifest.get('format') != ev.FORMAT or manifest.get('phase') != 'dev':
        raise ValueError('Selection provenance must be a frozen dev manifest, never confirmation')
    if any(manifest.get(k) != v for k, v in dict(game=game,
            config=cls.config_id, rules_version=cls.rules_version).items()):
        raise ValueError('Development game/config/rules mismatch')
    for bot in manifest['bots'].values():
        if _identity(bot['spec']) != bot['identity_sha256']:
            raise ValueError('Development bot identity mismatch')
    return manifest, hashlib.sha256(content).hexdigest()


def export_bundle(plan_file, output):
    """Copy an explicitly declared pool; paths in the plan are plan-relative.

    A game entry has bots, dev_manifest, status, recommended and optional research.
    Unknown training metadata is deliberately excluded from portable policy JSONs.
    """
    plan_file, output = Path(plan_file).resolve(), Path(output).absolute()
    plan = ev.read_json(plan_file)
    if set(plan) != {'games'} or not isinstance(plan['games'], dict) or not plan['games']:
        raise ValueError('Plan must contain a nonempty games mapping')
    if output.exists() or output.is_symlink():
        raise ValueError('Export requires a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.bot-bundle-', dir=output.parent)).resolve()
    try:
        games = {}
        for game, request in sorted(plan['games'].items()):
            cls = game_class(game)
            required = {'bots', 'dev_manifest', 'status', 'recommended'}
            if not isinstance(request, dict) or not required <= set(request) or set(request)-required-{'research'}:
                raise ValueError(f'{game}: expected bots, dev_manifest, status, recommended, optional research')
            if request['status'] not in ('selected', 'research'):
                raise ValueError('Game status must be selected or research')
            bot_file = ev.resolve_reference(request['bots'], plan_file.parent)
            dev_file = ev.resolve_reference(request['dev_manifest'], plan_file.parent)
            dev, dev_file_sha256 = _read_dev(dev_file, game)
            directory = staging/'games'/game
            directory.mkdir(parents=True)
            bots = ev.freeze_bots(bot_file, directory, game)
            recommended, research = request['recommended'], request.get('research', [])
            for names in (recommended, research):
                if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
                        or len(names) != len(set(names)) or not set(names) <= set(bots)):
                    raise ValueError('Recommended/research names must be unique members of the supplied pool')
            if set(recommended) & set(research):
                raise ValueError('A research-only bot cannot also be recommended')
            if request['status'] == 'selected' and not recommended:
                raise ValueError('A selected game requires an explicit recommended bot')
            entries, references = {}, {}
            for name, frozen in bots.items():
                if not NAME.fullmatch(name):
                    raise ValueError('Bot names must be short filesystem-safe identifiers')
                if (name not in dev['bots'] or frozen['identity_sha256'] !=
                        dev['bots'][name]['identity_sha256']):
                    raise ValueError(f'{game}/{name}: bot differs from the frozen dev candidate')
                spec = {k: deepcopy(v) for k, v in frozen['spec'].items() if k in POLICY_FIELDS}
                _check_guard(spec)
                if 'stochastic' in spec and type(spec['stochastic']) is not bool:
                    raise ValueError('stochastic must be boolean')
                model = None
                if frozen['model']:
                    copied = directory/frozen['model']['relative_path']
                    spec['path'] = '../'+frozen['model']['relative_path']
                    model = dict(path=str(copied.relative_to(staging)),
                                 sha256=frozen['model']['sha256'], bytes=copied.stat().st_size)
                target = directory/'bots'/f'{name}.json'
                write_json(target, spec)
                references[name] = 'bots/'+target.name
                entries[name] = dict(path=str(target.relative_to(staging)),
                    status='selected' if name in recommended else 'research' if name in research else 'retained',
                    identity_sha256=_identity(spec), dev_identity_sha256=frozen['identity_sha256'],
                    spec_source_sha256=[s['sha256'] for s in frozen['spec_sources']], model=model)
            write_json(directory/'bots.json', references)
            # The raw manifest includes absolute paths, original model locations
            # and potentially private annotations. Preserve its hash, not those fields.
            provenance = dict(format='boardbench_dev_provenance_v1', phase='dev',
                game=game, config=cls.config_id, rules_version=cls.rules_version,
                source_file_sha256=dev_file_sha256, source_manifest_sha256=dev['manifest_sha256'],
                seeds=dev['seeds'], bot_identities={n: dev['bots'][n]['identity_sha256'] for n in bots},
                source_code_sha256=dev['code']['content_sha256'])
            write_json(directory/'dev_provenance.json', provenance)
            games[game] = dict(config=cls.config_id, rules_version=cls.rules_version,
                status=request['status'], recommended=recommended, bots=entries,
                bots_file=str((directory/'bots.json').relative_to(staging)),
                dev_provenance=str((directory/'dev_provenance.json').relative_to(staging)))
        code = ev.code_identity()
        # Only portable code identity; historical source archives contain cluster
        # launchers and are intentionally not copied into a distributable bot pool.
        code = {k: code[k] for k in ('git_commit', 'tracked_dirty', 'content_sha256',
                                    'python', 'dependencies')}
        files = {str(p.relative_to(staging)): dict(sha256=ev.file_hash(p), bytes=p.stat().st_size)
                 for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = _signed(dict(format=FORMAT, games=games, files=files, code=code,
            selection='explicit dev choices supplied by caller; no ranking performed',
            selection_performed=False, source_archive=None))
        write_json(staging/'manifest.json', manifest)
        load_bundle(staging)
        if output.exists():
            raise ValueError('Output directory appeared during export')
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_bundle(directory, expected_sha256=None, verify_code=True):
    """Verify every artifact and return metadata; never load an unverified model."""
    root = Path(directory).resolve(strict=True)
    manifest = ev.read_json(_member(root, 'manifest.json'))
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported bot bundle format')
    _verify_signature(manifest)
    if expected_sha256 is not None and manifest['manifest_sha256'] != expected_sha256:
        raise ValueError('Bundle does not match the externally pinned manifest checksum')
    if verify_code and ev.code_identity()['content_sha256'] != manifest['code']['content_sha256']:
        raise ValueError('Code/data identity mismatch; use the recorded source version')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Bundle has missing or unexpected files')
    for relative, recorded in manifest['files'].items():
        path = _member(root, relative)
        if path.stat().st_size != recorded['bytes'] or ev.file_hash(path) != recorded['sha256']:
            raise ValueError(f'Bundle file checksum mismatch: {relative}')
    for game, entry in manifest['games'].items():
        cls = game_class(game)
        if (entry['config'], entry['rules_version']) != (cls.config_id, cls.rules_version):
            raise ValueError('Installed game/config/rules mismatch')
        for name, bot in entry['bots'].items():
            path = _member(root, bot['path'])
            spec = ev.read_json(path)
            _check_guard(spec)
            if (set(spec)-POLICY_FIELDS or spec.get('game') != game
                    or spec.get('config') != entry['config']
                    or spec.get('rules_version') != entry['rules_version']
                    or _identity(spec) != bot['identity_sha256']):
                raise ValueError(f'Portable policy identity mismatch: {game}/{name}')
            if bot['model']:
                model = bot['model']
                resolved = _member(root, spec['path'], path.parent)
                if (resolved != _member(root, model['path']) or spec['sha256'] != model['sha256']
                        or ev.file_hash(resolved) != model['sha256']):
                    raise ValueError('Portable model reference/checksum mismatch')
    return manifest


def load_bot(directory, game, name, seed=0, expected_sha256=None, verify_code=True):
    """Return the shared runtime policy with paths resolved relative to its JSON."""
    manifest = load_bundle(directory, expected_sha256, verify_code)
    root = Path(directory).resolve()
    bot = manifest['games'][game]['bots'][name]
    path = _member(root, bot['path'])
    spec = ev.read_json(path)
    if bot['model']:
        spec['path'] = str(_member(root, spec['path'], path.parent))
    return make_policy(game_class(game), spec, seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Copy an explicit development-selected pool')
    export.add_argument('--plan', required=True)
    export.add_argument('--output', required=True)
    verify = commands.add_parser('verify', help='Verify a moved bundle without training or inference')
    verify.add_argument('directory')
    verify.add_argument('--sha256', help='Trusted manifest content checksum from export')
    verify.add_argument('--allow-code-change', action='store_true',
                        help='Skip exact source hash only; game/config/rules and all files still checked')
    args = parser.parse_args(argv)
    manifest = (export_bundle(args.plan, args.output) if args.command == 'export' else
                load_bundle(args.directory, args.sha256, not args.allow_code_change))
    print(json.dumps(dict(manifest_sha256=manifest['manifest_sha256'],
        games={g: dict(status=v['status'], recommended=v['recommended'], bots=list(v['bots']))
               for g, v in manifest['games'].items()}), indent=2))


if __name__ == '__main__':
    main()
