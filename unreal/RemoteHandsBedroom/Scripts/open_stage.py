import unreal
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ok = les.load_level("/Game/Maps/RobotStage")
unreal.log_warning("REMOTE_HANDS open stage: %s" % ok)
