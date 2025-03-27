import sys
import csv
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
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

    #  remove wierd last row of all 1s, probably from extra delimiter + inversion
    inverted = inverted[:, :-1]
    # set outer border to 0 to allow clean mesh joining
    inverted = np.pad(
        inverted, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0
    )

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


def remove_scanlines(tiff_filepath, settings):
    img = Image.open(tiff_filepath)
    fast_axis = settings.get("fast_axis", 0)
    data = np.asarray(img)[1:-1, 1:-1]  # removes padding
    data = data.copy()
    d2axis = np.diff(data, n=1, axis=fast_axis)
    d2axis = abs(d2axis)
    val_min = d2axis.min()
    val_max = d2axis.max()
    # print(val_max)
    # Invert and  normalize to [0..1]
    #  Largest value -> 0, smallest -> 1
    # This just works the nicest with blender, trial and error ftw
    if val_min == val_max:
        inverted = np.zeros_like(d2axis, dtype=np.float32)
    else:
        inverted = (val_max - d2axis) / (val_max - val_min)
    val_min = inverted.min()
    val_max = inverted.max()
    # print(val_max, val_min)

    # Anything with a   derivative in the fast scanning axis than 80%
    inverted[inverted < 0.80] = 0
    rows, cols = np.where(inverted == 0)
    bad_rows = set(rows)

    if fast_axis == 0:
        # Loop over each flawed point
        for i, j in zip(rows, cols):
            neighbors = []

            # Check up neighbor (if exists)
            if i - 1 >= 0:
                if (i - 1) not in bad_rows:
                    neighbors.append(data[i - 1, j])
                else:
                    if i - 2 >= 0:
                        neighbors.append(data[i - 2, j])

            # Check down neighbor (if exists)
            if i + 1 < data.shape[0]:
                if (i + 1) not in bad_rows:
                    neighbors.append(data[i + 1, j])
                else:
                    if i + 2 < data.shape[0]:
                        neighbors.append(data[i + 2, j])

            if neighbors:
                data[i, j] = np.mean(neighbors)
    else:
        # Loop over each flawed point
        # just swapped i,j lol
        for j, i in zip(rows, cols):
            neighbors = []

            # Check up neighbor (if exists)
            if i - 1 >= 0:
                if (i - 1) not in bad_rows:
                    neighbors.append(data[i - 1, j])
                else:
                    if i - 2 >= 0:
                        neighbors.append(data[i - 2, j])

            # Check down neighbor (if exists)
            if i + 1 < data.shape[0]:
                if (i + 1) not in bad_rows:
                    neighbors.append(data[i + 1, j])
                else:
                    if i + 2 < data.shape[0]:
                        neighbors.append(data[i + 2, j])

            if neighbors:
                data[i, j] = np.mean(neighbors)

    data = np.pad(data, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0)
    img = Image.fromarray(data, mode="F")
    output_path = Path("C:/temp/outputtiff.tiff")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="TIFF")
    # print(d2axis)
    # adds padding back

    return


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


def plane_level(tiff_filepath):
    img = Image.open(tiff_filepath)
    data = np.asarray(img)[1:-1, 1:-1]  # removes padding
    data = data.copy()

    # Create coordinate arrays for the image dimensions
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X1, X2 = np.meshgrid(x, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(3, 1, 1, projection="3d")
    # jet = plt.get_cmap("jet")

    # Normalize the height data for display
    Y = data

    # Plot the initial topological surface
    # ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=jet, linewidth=0)

    # Regression
    X = np.hstack(
        (
            np.reshape(X1, (data.shape[1] * data.shape[0], 1)),
            np.reshape(X2, (data.shape[1] * data.shape[0], 1)),
        )
    )
    X = np.hstack((np.ones((data.shape[1] * data.shape[0], 1)), X))
    YY = np.reshape(Y, (data.shape[1] * data.shape[0], 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (data.shape[0], data.shape[1]))

    # ax = fig.add_subplot(3, 1, 2, projection="3d")
    # ax.plot_surface(X1, X2, plane)
    # ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=jet, linewidth=0)

    # Subtraction
    data = Y - plane
    # ax = fig.add_subplot(3, 1, 3, projection="3d")
    # ax.plot_surface(X1, X2, Y_sub, rstride=1, cstride=1, cmap=jet, linewidth=0)

    # plt.savefig("subtracted.png")

    if data.min() < 0:
        data = data + abs(data.min())
    # set outer border to 0 to allow clean mesh joining
    data = np.pad(data, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0)

    inverted_float32 = data.astype(np.float32)

    img = Image.fromarray(inverted_float32, mode="F")
    output_path = Path("C:/temp/outputtiff.tiff")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="TIFF")

    return


class SettingsWidget(QWidget):
    generate_stl_requested = pyqtSignal(dict)
    plane_level_signal = pyqtSignal()
    remove_scanlines_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_box = QGroupBox("Settings", self)
        self.form_layout = QFormLayout()
        self.setMinimumWidth(200)
        self.form_layout.setContentsMargins(8, 8, 8, 8)
        self.form_layout.setSpacing(10)
        self.iterations_label = QLabel("Scan Heal Iterations:")
        self.iterations_input = QDoubleSpinBox()
        self.iterations_input.setRange(1, 10)
        self.iterations_input.setSingleStep(1)
        self.iterations_input.setDecimals(0)
        self.iterations_input.setValue(1)

        self.plane_level_button = QPushButton("Plane Level")
        self.remove_scanlines_button = QPushButton("Heal Scanlines")
        self.scan_axis_label = QLabel("Fast Scan Axis:")
        self.scan_axis = QComboBox()
        self.scan_axis.addItem("Y")
        self.scan_axis.addItem("X")

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
        self.form_layout.addRow("", self.plane_level_button)
        self.form_layout.addRow(self.scan_axis_label, self.scan_axis)
        self.form_layout.addRow(self.iterations_label, self.iterations_input)
        self.form_layout.addRow("", self.remove_scanlines_button)
        self.form_layout.addRow("", self.generate_button)
        self.plane_level_button.clicked.connect(self.plane_level_requested)
        self.remove_scanlines_button.clicked.connect(self.scanlines_requested)
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

    def plane_level_requested(self):
        self.plane_level_signal.emit()

    def scanlines_requested(self):
        iterations = int(self.iterations_input.value())
        self.remove_scanlines_signal.emit(iterations)

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
            "fast_axis": self.scan_axis.currentIndex(),
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
        self.settings_widget.plane_level_signal.connect(self.handle_plane_level)
        self.settings_widget.remove_scanlines_signal.connect(
            self.handle_scanline_removal
        )
        self.file_drop_widget.file_loaded.connect(
            self.settings_widget.set_default_filename
        )

    def handle_scanline_removal(self, settings):
        if self.file_drop_widget.current_tiff is None:
            self.file_drop_widget.show_error(
                "No file loaded! Please drop a file first."
            )
            return

    def handle_plane_level(self):
        if self.file_drop_widget.current_tiff is None:
            self.file_drop_widget.show_error(
                "No file loaded! Please drop a file first."
            )
            return

        plane_level(self.file_drop_widget.current_tiff)
        self.file_drop_widget.load_preview(self.file_drop_widget.current_tiff)

    def handle_scanline_removal(self, iterations):
        if self.file_drop_widget.current_tiff is None:
            self.file_drop_widget.show_error(
                "No file loaded! Please drop a file first."
            )
            return

        for _ in range(iterations):
            remove_scanlines(self.file_drop_widget.current_tiff, {})

        self.file_drop_widget.load_preview(self.file_drop_widget.current_tiff)

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
