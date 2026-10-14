"""Register the paper's names with the locally frozen Ultralytics parser.

The original parser dispatches on object identity. Rebinding its three existing
names lets clean names and historical YAML names use the same parser branches;
the original modules remain importable for historical pickle checkpoints.
"""


def register_modules():
    """Register idempotently, without modifying any installed Ultralytics files."""
    from ultralytics.nn import tasks

    from .modules.attention import PfAAM
    from .modules.backbone import shufflenetv2_05
    from .modules.comparators import TripletAttention
    from .modules.mlla import C2PSAMLLA
    from .modules.neck import SRSM

    tasks.shufflenetv2_05 = shufflenetv2_05
    tasks.ESPBlock = SRSM
    tasks.SRSM = SRSM
    tasks.LSKA = PfAAM
    tasks.PfAAM = PfAAM
    # This comparator preserves channel count and uses the parser's generic
    # no-channel-injection branch; its YAML therefore passes no channel args.
    tasks.TripletAttention = TripletAttention
    # The frozen nano comparator supplies actual [128,128] channel arguments;
    # the generic parser must not apply width scaling or repeat insertion again.
    tasks.C2PSAMLLA = C2PSAMLLA
