"""Per-target verification flags.

tau is chosen on validation-region targets only (see verify/run.py, threshold log). We read
"the top-3 event has |delta| >= tau" as: at least 2 of the explainer's top-3 events do.
  INSUFFICIENT  no candidate has |delta| >= tau (redundant evidence or a flat model)
  CONSISTENT    >= 2 of the explainer's top-3 have |delta| >= tau
  DISAGREEMENT  otherwise: the explainer calls events important that replay says are not
"""


def flag(ranked, deltas, tau, top=3, need=2):
  strong = [e for e in deltas if abs(deltas[e]) >= tau]
  top_events = ranked[:top]
  hits = [e for e in top_events if abs(deltas[e]) >= tau]
  if not strong:
    return 'INSUFFICIENT', (f'No candidate moves the score by at least tau={tau:.4f} when removed '
                            f'(max |delta|={max(abs(d) for d in deltas.values()):.4f}); evidence is '
                            f'redundant or the model is flat here.')
  if len(hits) >= min(need, len(top_events)):
    return 'CONSISTENT', (f'{len(hits)} of the explainer\'s top-{len(top_events)} events move the score '
                          f'by >= tau={tau:.4f} when removed ({len(strong)} candidates do overall).')
  return 'DISAGREEMENT', (f'Only {len(hits)} of the explainer\'s top-{len(top_events)} events move the '
                          f'score by >= tau={tau:.4f}; replay finds {len(strong)} other candidates that do. '
                          f'The explainer calls events important that replay says are not.')
