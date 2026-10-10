import os

# Rendering (--render) opens a MuJoCo viewer window through GLFW. Under a Wayland session the glfw package
# loads its Wayland build, which draws window decorations with libdecor's GTK plugin; on GNOME 50 that GTK
# aborts on a removed settings key ("Settings schema 'org.gnome.settings-daemon.plugins.xsettings' does
# not contain a key named 'antialiasing'"). The X11 build (through XWayland) does not load GTK. Set here,
# before anything imports glfw (gymnasium_robotics does on import); an explicit setting is kept.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
