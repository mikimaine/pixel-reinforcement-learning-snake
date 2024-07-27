import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QFormLayout, QPushButton, QMessageBox

from trainingagent import train


class GameSettings:
    def __init__(self):
        self.settings = {
            'speed': {'value': 20, 'type': 'int'},
            'state_size': {'value': 11, 'type': 'int'},
            'hidden_layer_size': {'value': 256, 'type': 'int'},
            'action_size': {'value': 3, 'type': 'int'},
            'max_memory': {'value': 100, 'type': 'int'},
            'epsilon_decay': {'value': 80, 'type': 'int'},
            'learning_rate': {'value': 0.001, 'type': 'float'},
            'batch_size': {'value': 1000, 'type': 'int'},
            'gamma': {'value': 0.9, 'type': 'float'}
        }


class SettingScreen(QWidget):
    def __init__(self, settings, start_game_callback):
        super().__init__()
        self.train_agent_button = None
        self.game_settings = settings
        self.start_game_callback = start_game_callback
        self.input_fields = {}
        self.ui()

    def ui(self):
        layout = QFormLayout()
        self.setMinimumSize(400, 300)  # Ensure UI elements are not cramped.

        for setting, info in self.game_settings.settings.items():
            input_field = QLineEdit(str(info['value']))
            self.input_fields[setting] = input_field
            layout.addRow(f'{setting.replace("_", " ").title()}', input_field)

        self.train_agent_button = QPushButton('Train Agent', self)
        self.train_agent_button.clicked.connect(self.train_agent)
        layout.addWidget(self.train_agent_button)

        self.setLayout(layout)
        self.setWindowTitle('Game Settings')
        self.show()

    def train_agent(self):
        error = False
        for setting, input_field in self.input_fields.items():
            text_value = input_field.text()
            setting_type = self.game_settings.settings[setting]['type']
            try:
                if setting_type == 'float':
                    value = float(text_value)
                elif setting_type == 'int':
                    value = int(text_value)
                else:
                    value = text_value
                self.game_settings.settings[setting]['value'] = value
            except ValueError:
                QMessageBox.warning(self, "Invalid Input",
                                    f"Please enter a valid number for {setting.replace('_', ' ').title()}.")
                input_field.setText(str(self.game_settings.settings[setting]['value']))  # Reset to the last valid value
                error = True
        if not error:
            self.start_game_callback(self.game_settings)


def train_agent(game_settings):
    print(f'Game settings: {game_settings.settings}')
    train()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game_settings = GameSettings()
    ap = SettingScreen(game_settings, train_agent)
    sys.exit(app.exec_())
