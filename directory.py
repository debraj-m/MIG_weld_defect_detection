import os

def print_structure(root, max_files=5):
    lines = []

    def walk(path, prefix=""):
        items = sorted(os.listdir(path))
        file_count = 0

        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                lines.append(f"{prefix}|-- {item}/")
                walk(full_path, prefix + "    ")
            else:
                file_count += 1
                if file_count <= max_files:
                    lines.append(f"{prefix}|-- {item}")
                elif file_count == max_files + 1:
                    lines.append(f"{prefix}|-- ... ({len(items) - max_files} more files)")

    walk(root)
    return "\n".join(lines)

# Set your dataset path here
dataset_path = r"C:\Users\debra\Desktop\CODE\Dataset"
output_text = print_structure(dataset_path)

# Save to a text file
with open("folder_structure.txt", "w") as f:
    f.write(output_text)

# OPTIONAL: Convert to PDF
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Courier", size=10)

for line in output_text.split("\n"):
    pdf.cell(200, 5, txt=line, ln=True)

pdf.output("folder_structure.pdf")
