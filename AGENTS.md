# BoardGameBench repository workflow

## Git delivery

- The user explicitly requests a push after every completed project update.
  Commit coherent, verified changes and push to the configured upstream before
  handing off; do not push every temporary edit or incomplete experiment batch.
- Verify the remote branch SHA after pushing. If synchronization is blocked,
  report the local commit and the blocker rather than claiming publication.
- Preserve the explicit repository identity: `guoruihan <guoruihan.sansi@163.com>`.
  Do not alter global Git identity, force-push, or rewrite shared history.
- Inspect status and stage only relevant files. Preserve unrelated user changes.
  Generated weights, trajectories and run directories remain outside Git unless
  the user explicitly requests a separate artifact release.
- Update `CHANGES.md` for published engineering/documentation changes and
  `CHANGELOG.md` when the research method or evaluation milestone changes.

## Private service integration

- Keep service-specific endpoint addresses, authentication configuration,
  invocation scripts and private instructions out of Git. Store them outside
  the repository or in ignored local directories; never force-add them.
- Public integration code may expose a generic interface. Check staged content
  for private connection details before every related commit and push.
