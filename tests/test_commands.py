from safegait.commands import Task, parse_command

OBJECTS = ["red_cube", "blue_cube", "green_box"]


def test_full_name():
    assert parse_command("Please pick up the red cube.", OBJECTS) == Task("pick", "red_cube")


def test_unique_noun_is_enough():
    assert parse_command("reach the box", OBJECTS) == Task("reach", "green_box")


def test_ambiguous_object_is_rejected():
    assert parse_command("grab the cube", OBJECTS) is None


def test_unknown_verb_or_object_is_rejected():
    assert parse_command("dance with the red cube", OBJECTS) is None
    assert parse_command("pick up the purple sphere", OBJECTS) is None
