"""app.py — dashboard arg definitions stay in sync with the stage registry.

STAGE_ARG_DEFS carries UI presentation (labels, hints, defaults) that the
registry doesn't, so it can't be derived — this test makes the cross-check in
app.py a hard failure instead of a startup warning: adding or renaming an Opt
in pipeline/stages.py without updating the dashboard (or _UNEXPOSED_OPTS)
breaks CI rather than silently shipping a UI/CLI mismatch.
"""

import app


def test_stage_arg_defs_match_registry():
    assert app._stage_arg_drift() == []


def test_unexposed_opts_exist_in_registry():
    # _UNEXPOSED_OPTS must reference real registry opts, or it silently rots.
    for stage, attrs in app._UNEXPOSED_OPTS.items():
        reg = app._REGISTRY[stage]
        registry_attrs = {o.attr for o in reg.opts}
        assert attrs <= registry_attrs, (
            f"{stage}: _UNEXPOSED_OPTS names opts not in the registry: "
            f"{sorted(attrs - registry_attrs)}"
        )
