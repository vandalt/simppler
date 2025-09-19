import simppler


def test_simppler_version():
    from importlib.metadata import version

    # For editable installs, you might just need to re-install if this fails
    assert simppler.__version__ == version("simppler")
