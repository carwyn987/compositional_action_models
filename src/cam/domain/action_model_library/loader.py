"""Load a skill's symbolic action model in a chosen representation format.

Each format has a directory next to this file holding one file per skill,
e.g. pddl/pickup.pddl.
"""

from importlib.resources import files

from cam.domain.operators.pddl_parser import parse_operator

# format -> (file extension, parser)
SYMBOLIC_ACTION_MODEL_FORMATS = {
    "pddl": (".pddl", parse_operator),
}


def load_symbolic_action_model(skill_name: str, symbolic_action_model_format: str):
    if symbolic_action_model_format not in SYMBOLIC_ACTION_MODEL_FORMATS:
        raise ValueError(
            f"unknown symbolic action model format {symbolic_action_model_format!r}; "
            f"known: {sorted(SYMBOLIC_ACTION_MODEL_FORMATS)}"
        )
    extension, parse = SYMBOLIC_ACTION_MODEL_FORMATS[symbolic_action_model_format]
    path = files(__package__) / symbolic_action_model_format / f"{skill_name}{extension}"
    if not path.is_file():
        raise FileNotFoundError(f"no {symbolic_action_model_format} action model for {skill_name!r}: {path}")
    return parse(path.read_text())
