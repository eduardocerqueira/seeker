#date: 2025-07-25T17:15:00Z
#url: https://api.github.com/gists/45d617587a8ea2bc3cdef776bc4187a3
#owner: https://api.github.com/users/MichaelGift

if __name__ == "__main__":
    editor = TextEditor()
    manager = CommandManager()

    print("--- Performing actions ---")
    manager.execute_command(TypeCommand(editor, 'H'))
    manager.execute_command(TypeCommand(editor, 'e'))
    manager.execute_command(TypeCommand(editor, 'l'))
    manager.execute_command(TypeCommand(editor, 'l'))
    manager.execute_command(TypeCommand(editor, 'o'))
    print(f"Current Editor Content: '{editor.get_content()}'")

    print("\n--- Undoing actions ---")
    manager.undo() # Undo 'o'
    manager.undo() # Undo 'l'
    print(f"Current Editor Content: '{editor.get_content()}'")

    print("\n--- Redoing actions ---")
    manager.redo() # Redo 'l'
    print(f"Current Editor Content: '{editor.get_content()}'")

    print("\n--- Performing new action after undo ---")
    manager.execute_command(TypeCommand(editor, ' W'))
    print(f"Current Editor Content: '{editor.get_content()}'")

    print("\n--- Trying to Redo (shouldn't work after new command) ---")
    manager.redo()