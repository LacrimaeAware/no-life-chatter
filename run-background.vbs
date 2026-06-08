' Starts the bot completely hidden (no window). Also used by auto-start at login.
' Double-click this any time to launch the bot in the background.
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = projectDir
' 0 = hidden window, False = don't wait for it to finish
sh.Run "cmd /c """ & projectDir & "\_bot-loop.bat""", 0, False
