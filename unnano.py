import sys
import csv
import pymesh
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import bpy
import open3d as o3d
import math


def csv_to_32bit_float_inverted_tiff(input_filepath):
    data_list = []
    input_filepath = Path(input_filepath).resolve()
    # read semicolon delimited csv
    with open(input_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            # skip empty rows
            if not row or all(cell.strip() == "" for cell in row):
                continue

            float_row = []
            for cell in row:
                val = cell.strip()
                if val == "":
                    # Replace empty cells with 0.0
                    float_row.append(0.0)
                else:
                    float_row.append(float(val))
            data_list.append(float_row)

    #   32bit here and at tiff to preserve detail
    data = np.array(data_list, dtype=np.float32)

    val_min = data.min()
    val_max = data.max()

    # Invert and  normalize to [0..1]
    #  Largest value -> 0, smallest -> 1
    # This just works the nicest with blender, trial and error ftw
    if val_min == val_max:
        inverted = np.zeros_like(data, dtype=np.float32)
    else:
        inverted = (val_max - data) / (val_max - val_min)

    # set outer border to 0 to allow clean mesh joining
    if inverted.shape[0] > 1:
        inverted[0, :] = 0.0
        inverted[-1, :] = 0.0
    if inverted.shape[1] > 1:
        inverted[:, 0] = 0.0
        inverted[:, -1] = 0.0

    inverted_float32 = inverted.astype(np.float32)

    img = Image.fromarray(inverted_float32, mode="F")
    output_path = Path("C:/temp/outputtiff.tiff")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="TIFF")
    return output_path


def generate_stl(tiff_filepath, settings):
    """
    Generates an STL from a TIFF displacement map using Blender (Fully Headless).
    Only the top face of a cube is processed with > 500^2 faces

    """
    # Convert paths to Pathlib objects
    tiff_path = Path(tiff_filepath).resolve()
    output_path = Path(settings.get("output_path", "C:/temp/output.obj")).resolve()
    SampleName = settings.get("sample_name", "Example")
    FontSize = settings.get("font_size", 1.00)
    blend_file_path = "model_template/scanModel.blend"
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    rotation_degrees = settings.get("rotation_degrees", 0)
    # Swap displacement texture for the new one
    obj = bpy.data.objects["SampleMesh"]
    displacement_modifier = obj.modifiers.get("AFM_Scan")

    if displacement_modifier:
        texture_slot = displacement_modifier.texture
        if texture_slot:
            new_image = bpy.data.images.load(str(tiff_path))
            texture_slot.image = new_image

            texture_slot.image.colorspace_settings.is_data = True  # load as noncolor

            print(f"Displacement texture changed to: {new_image.name}")
        else:
            print("No texture found in the displacement modifier.")
    else:
        print("No displacement modifier found on the object.")

    angle_radians = math.radians(rotation_degrees)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler[2] += angle_radians
    print(f"Applied a rotation of {rotation_degrees} degrees around Z-axis.")
    # Change Text
    text_obj = bpy.data.objects.get("SampleName")
    if text_obj and text_obj.type == "FONT":
        text_obj.data.body = SampleName
        text_obj.data.size = FontSize
        print(f"Text object updated: {text_obj.data.body}")
    else:
        print("Text object not found or the object is not a text object.")

    # Save the modified .blend file
    output_blend_file = "modified_template.blend"
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)
    print(f"Modified blend file saved to {output_blend_file}")

    # Save the STL
    bpy.ops.wm.stl_export(filepath=str(output_path))
    print(f"Stl saved to {output_path}")
    return output_path


class SettingsWidget(QWidget):
    generate_stl_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_box = QGroupBox("Settings", self)
        self.form_layout = QFormLayout()
        self.setMinimumWidth(200)
        self.form_layout.setContentsMargins(8, 8, 8, 8)
        self.form_layout.setSpacing(10)

        self.height_label = QLabel("Height Scale:")
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.1, 100.0)
        self.height_input.setSingleStep(0.1)
        self.height_input.setDecimals(2)
        self.height_input.setValue(1.0)

        self.font_size_label = QLabel("Font Size:")
        self.font_size_input = QDoubleSpinBox()
        self.font_size_input.setRange(0.1, 100.0)
        self.font_size_input.setSingleStep(0.1)
        self.font_size_input.setDecimals(2)
        self.font_size_input.setValue(1.0)

        self.rotation_label = QLabel("Rotation (degrees):")
        self.rotation_input = QDoubleSpinBox()
        self.rotation_input.setRange(-360, 360)
        self.rotation_input.setSingleStep(90)
        self.rotation_input.setDecimals(2)
        self.rotation_input.setValue(0)

        self.output_folder_label = QLabel("Output Folder:")
        self.output_folder_line_edit = QLineEdit()
        self.output_folder_line_edit.setPlaceholderText("Select an output folder...")
        self.output_folder_button = QPushButton("Browse")
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.output_folder_line_edit, stretch=1)
        folder_layout.addWidget(self.output_folder_button)

        self.output_file_label = QLabel("Output File Name:")
        self.output_file_line_edit = QLineEdit()
        self.output_file_line_edit.setPlaceholderText("(Optional)")

        self.SampleName_label = QLabel("Sample Label:")
        self.SampleName_line_edit = QLineEdit()
        self.SampleName_line_edit.setPlaceholderText("Example")

        self.generate_button = QPushButton("Generate STL")

        self.form_layout.addRow(self.height_label, self.height_input)
        self.form_layout.addRow(self.output_folder_label, folder_layout)
        self.form_layout.addRow(self.output_file_label, self.output_file_line_edit)
        self.form_layout.addRow(self.SampleName_label, self.SampleName_line_edit)
        self.form_layout.addRow(self.font_size_label, self.font_size_input)
        self.form_layout.addRow(self.rotation_label, self.rotation_input)
        self.form_layout.addRow("", self.generate_button)

        self.group_box.setLayout(self.form_layout)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.group_box)

        self.output_folder_button.clicked.connect(self.select_output_folder)
        self.generate_button.clicked.connect(self.on_generate_stl)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.output_folder_line_edit.setText(folder)

    def on_generate_stl(self):
        settings = self.get_settings()
        if not settings.get("output_path"):
            self.output_folder_line_edit.setPlaceholderText("Required")
            return
        self.generate_stl_requested.emit(settings)

    def get_settings(self):
        folder = self.output_folder_line_edit.text().strip()
        file_name = self.output_file_line_edit.text().strip()
        sample_name = self.SampleName_line_edit.text().strip()
        if not sample_name:
            sample_name = "Example"
        if not file_name:
            file_name = "default.stl"
        if not file_name.lower().endswith(".stl"):
            file_name += ".stl"
        output_path = str(Path(folder) / file_name) if folder else ""
        return {
            "height_scale": self.height_input.value(),
            "output_path": output_path,
            "sample_name": sample_name,
            "font_size": self.font_size_input.value(),
            "rotation_degrees": self.rotation_input.value(),
        }

    def set_default_filename(self, default_name):
        if not self.output_file_line_edit.text().strip():
            base = Path(default_name).stem
            self.output_file_line_edit.setText(f"{base}.stl")


class FileDropWidget(QWidget):
    file_loaded = pyqtSignal(str)

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setObjectName("FileDropWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.current_tiff = None
        self.original_pixmap = None
        self.setMinimumSize(400, 400)
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        valid = False
        for f in files:
            file_path = Path(f)
            if file_path.suffix.lower() == ".csv":
                file_path = csv_to_32bit_float_inverted_tiff(file_path)
                self.load_preview(file_path)
                valid = True
                break
        if not valid:
            self.show_error("Invalid file type! Please use .csv")

    def load_preview(self, file_path):
        self.current_tiff = file_path
        pixmap = QPixmap(str(file_path))
        if pixmap.isNull():
            self.show_error("Could not load image preview.")
            return
        self.original_pixmap = pixmap
        self.update_preview_pixmap()
        self.file_loaded.emit(file_path.name)

    def update_preview_pixmap(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)
            self.label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview_pixmap()

    def show_error(self, message):
        self.original_pixmap = None
        self.current_tiff = None
        self.label.setPixmap(QPixmap())
        self.label.setText(message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UnNano - AFM 3D Model Generator")

        central_widget = QWidget(self)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.file_drop_widget = FileDropWidget(
            "Drag and Drop a .csv from a Thorlabs AFM", self
        )
        self.settings_widget = SettingsWidget()

        layout.addWidget(self.file_drop_widget, stretch=3)
        layout.addWidget(self.settings_widget, stretch=2)
        self.setCentralWidget(central_widget)

        self.settings_widget.generate_stl_requested.connect(self.handle_generate_stl)
        self.file_drop_widget.file_loaded.connect(
            self.settings_widget.set_default_filename
        )

    def handle_generate_stl(self, settings):
        if self.file_drop_widget.current_tiff is None:
            self.file_drop_widget.show_error(
                "No file loaded! Please drop a file first."
            )
            return
        stl_path = generate_stl(self.file_drop_widget.current_tiff, settings)
        mesh = o3d.io.read_triangle_mesh(str(stl_path.resolve()))
        mesh = mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"{str(Path(stl_path).stem)}",
            left=1000,
            top=200,
            width=800,
            height=650,
        )
        self.file_drop_widget.label.setText("STL generated and loaded.")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet(
        """
        QWidget#FileDropWidget {
            background-color: #e0f7fa;
            border: 2px solid #004d40;
            border-radius: 8px;
        }
        QLabel {
            color: #004d40;
            font-size: 12px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #aaa;
            border-radius: 5px;
            margin-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QPushButton {
            background-color: #80deea;
            color: #004d40;
            border: 1px solid #004d40;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: #b2ebf2;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #80deea;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #004d40;
            width: 12px;
            margin: -5px 0;
            border-radius: 4px;
        }
        QDoubleSpinBox, QLineEdit {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 2px;
        }
        """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
