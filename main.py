import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QFormLayout, QPushButton, QMessageBox, QHBoxLayout

from test_agent import test
from trainingagent import train
from settings import GameSettings
from dataclasses import asdict


class SettingScreen(QWidget):
    def __init__(self, settings, start_game_callback, test_game_callback):
        super().__init__()
        self.default_values = asdict(settings)
        self.game_settings = settings
        self.start_game_callback = start_game_callback
        self.test_game_callback = test_game_callback
        self.input_fields = {}
        self.ui()

    def ui(self):
        layout = QFormLayout()
        self.setMinimumSize(400, 300)

        # Automatically creating fields based on game_settings attributes
        for setting in self.default_values:
            input_field = QLineEdit(str(getattr(self.game_settings, setting)))
            self.input_fields[setting] = input_field
            default_value = self.default_values[setting]
            # Adding label with default value shown
            label_text = f'{setting.replace("_", " ").title()} (default: {default_value})'
            layout.addRow(label_text, input_field)

        train_button = QPushButton('Train Agent', self)
        train_button.clicked.connect(self.train_agent)
        test_button = QPushButton('Test Agent', self)
        test_button.clicked.connect(self.test_agent)

        # Layout both buttons side by side
        button_layout = QHBoxLayout()
        button_layout.addWidget(train_button)
        button_layout.addWidget(test_button)
        layout.addRow(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('Game Settings')
        self.show()

    def train_agent(self):
        self.handle_agent_action(self.start_game_callback)

    def test_agent(self):
        self.handle_agent_action(self.test_game_callback)

    def handle_agent_action(self, action_callback):
        error = False
        for setting, input_field in self.input_fields.items():
            text_value = input_field.text()
            try:
                value = float(text_value) if '.' in text_value else int(text_value)
                setattr(self.game_settings, setting, value)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input",
                                    f"Please enter a valid number for {setting.replace('_', ' ').title()}.")
                input_field.setText(str(getattr(self.game_settings, setting)))  # Reset to the last valid value
                error = True
        if not error:
            action_callback(self.game_settings)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game_settings = GameSettings()
    ap = SettingScreen(game_settings, train, test)
    sys.exit(app.exec_())
