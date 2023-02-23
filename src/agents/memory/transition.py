from collections import namedtuple

Transition = namedtuple(
  'Transition', (
    'state',
    'action',
    'log_prob',
    'state_value',
    'reward',
    'is_done'
  )
)
