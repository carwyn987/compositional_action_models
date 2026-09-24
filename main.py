"""Entry point: python main.py --skills pickup [more ...] --symbolic-action-model-format pddl"""

import argparse

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.skills.registry import SKILL_REGISTRY, build_skill


def parse_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", nargs="+", required=True, choices=sorted(SKILL_REGISTRY))
    parser.add_argument(
        "--symbolic-action-model-format",
        default="pddl",
        choices=sorted(SYMBOLIC_ACTION_MODEL_FORMATS),
    )
    return vars(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    skills = [build_skill(skill_name, config) for skill_name in config["skills"]]
    for skill in skills:
        print(skill.symbolic_action_model)


if __name__ == "__main__":
    main()
