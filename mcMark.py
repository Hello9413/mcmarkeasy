import cv2
import numpy as np
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button, Label, Frame
import json
import csv
import os

class PDFOMRScanner:
    def __init__(self, pdf_path):
        print("Converting PDF... please wait.")
        self.pages = convert_from_path(pdf_path, dpi=200)
        self.current_page_index = 0
        self.all_results = [] # To store answers for the CSV
        self.config_file = "omr_config.json"

        # Setup GUI
        self.root = tk.Tk()
        self.root.geometry("450x850")
        self.root.title("Advanced OMR Grid Control")

        # --- Buttons Frame (at the bottom, non-scrollable) ---
        btn_frame = Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill='x', pady=10, padx=10)
        Button(btn_frame, text="Save Calibration", command=self.save_settings, bg="#28A745", fg="black").pack(side=tk.LEFT, expand=True, padx=5)
        Button(btn_frame, text="Confirm & Next Page", command=self.next_page, bg="#007BFF", fg="black").pack(side=tk.RIGHT, expand=True, padx=5)

        # --- Create a scrollable frame for the sliders ---
        container = Frame(self.root)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        Label(scrollable_frame, text="Adjust sliders to align boxes. Settings save automatically.", font=("Arial", 10, "bold")).pack(pady=5)

        # Helper function to create sliders
        self.sliders = {}
        def create_slider(key, label, from_val, to_val, default_val):
            var = tk.IntVar(value=default_val)
            Scale(scrollable_frame, from_=from_val, to_=to_val, orient=HORIZONTAL,
                  label=label, variable=var, command=self.update).pack(fill='x', padx=15)
            self.sliders[key] = var
            return var

        # Create Sliders
        self.thresh_val = create_slider("thresh", "Pencil Darkness Threshold", 50, 255, 150)
        self.total_q    = create_slider("total_q", "Total Questions", 1, 100, 50)

        Label(scrollable_frame, text="--- Position & Size ---").pack(pady=5)
        self.start_x    = create_slider("start_x", "1. Start X (Left Margin)", 0, 800, 100)
        self.start_y    = create_slider("start_y", "2. Start Y (Top Margin)", 0, 900, 100)
        self.box_w      = create_slider("box_w", "3. Box Width", 10, 100, 30)
        self.box_h      = create_slider("box_h", "4. Box Height", 10, 100, 15)

        Label(scrollable_frame, text="--- Spacing (Gaps) ---").pack(pady=5)
        self.opt_gap    = create_slider("opt_gap", "5. Gap between A-B-C-D", 0, 100, 25)
        self.q_gap      = create_slider("q_gap", "6. Vertical Gap (Row to Row)", 0, 100, 20)
        self.block_gap  = create_slider("block_gap", "7. Block Gap (Extra space after Q5)", 0, 200, 40)
        self.col_gap    = create_slider("col_gap", "8. Column Gap (Distance to Q26)", 200, 1200, 500)
        self.col_gap_main = create_slider("col_gap_main", "9. Main Gap (to Q51+)", 200, 1200, 700)

        Label(scrollable_frame, text="--- Header Identification ---").pack(pady=5)
        self.header_y   = create_slider("header_y", "Header Start Y", 0, 600, 50)
        self.level_x    = create_slider("level_x", "Level X (1-6)", 0, 1200, 50)
        self.class_x    = create_slider("class_x", "Class X (CH,FA,HO,JU)", 0, 1400, 150)
        self.c_no_x     = create_slider("c_no_x", "Class No X (00)", 0, 1600, 300)
        self.category_x = create_slider("category_x", "Category X (A-H)", 0, 1600, 450)
        self.h_row_gap  = create_slider("h_row_gap", "Header Row Gap", 0, 50, 5)
        self.h_col_gap  = create_slider("h_col_gap", "Header Col Gap", 0, 50, 10)

        # Load saved settings if they exist
        self.load_settings()

        self.update()
        self.root.mainloop()

    # --- SAVE / LOAD SETTINGS LOGIC ---
    def save_settings(self):
        settings = {key: var.get() for key, var in self.sliders.items()}
        with open(self.config_file, 'w') as f:
            json.dump(settings, f)
        print("Calibration saved successfully!")

    def load_settings(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                settings = json.load(f)
                for key, val in settings.items():
                    if key in self.sliders:
                        self.sliders[key].set(val)
            print("Loaded previous calibration settings.")

    # --- IMAGE ALIGNMENT LOGIC ---
    def align_image(self, img):
        # Get the width and height of the raw scan
        img_h, img_w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Using your successful threshold of 110!
        _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_markers = []
        MIN_AREA = 800
        MAX_AREA = 60000

        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h

                # Filter for Squares: Aspect ratio close to 1.0
                if 0.8 <= aspect_ratio <= 1.2:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = float(area) / hull_area
                        if solidity > 0.7:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])

                                # --- START OF SAFE ZONE FILTER ---
                                # Convert the dot's position to a percentage (0.0 to 1.0)
                                px = cx / img_w
                                py = cy / img_h

                                # Check X axis (Left side is ~31%, Right side is ~92%)
                                # Allow anything between 15%-45% OR 75%-98%
                                valid_x = (0.15 < px < 0.45) or (0.75 < px < 0.98)

                                # Check Y axis (Top is ~33%, Bottom is ~94%)
                                # Allow anything between 20%-45% OR 80%-99%
                                valid_y = (0.31 < py < 0.45) or (0.80 < py < 0.99)

                                # If the dot is NOT inside the safe boxes, ignore it completely!
                                if not (valid_x and valid_y):
                                    continue
                                # --- END OF SAFE ZONE FILTER ---

                                # If it passed the safe zone check, save it!
                                valid_markers.append((cx, cy))

        if len(valid_markers) != 4:
            print(f"Warning: Found {len(valid_markers)} markers instead of 4. Skipping alignment.")

            # Draw red circles on everything it thinks is a marker
            debug_img = img.copy()
            for pt in valid_markers:
                cv2.circle(debug_img, pt, 25, (0, 0, 255), -1)

            # Shrink it so it fits on your screen
            h, w = debug_img.shape[:2]
            debug_resized = cv2.resize(debug_img, (w // 2, h // 2))

            # Show the image and PAUSE the program until you press a key
            window_name = f"ERROR ON PAGE! Found {len(valid_markers)} dots. Press any key to continue."
            cv2.imshow(window_name, debug_resized)
            cv2.waitKey(0)  # Pauses here so you can inspect the red dots!
            cv2.destroyWindow(window_name)

            return img

        # --- DON'T FORGET YOUR WARPING CODE! ---
        # (Make sure your sorting and cv2.warpPerspective code is still
        # sitting right here at the bottom of the function!)
        # Sort the markers
        valid_markers = sorted(valid_markers, key=lambda x: x[1])
        top_markers = sorted(valid_markers[:2], key=lambda x: x[0])
        bottom_markers = sorted(valid_markers[-2:], key=lambda x: x[0])

        tl, tr = top_markers[0], top_markers[1]
        bl, br = bottom_markers[0], bottom_markers[1]

        ordered_pts = np.array([tl, tr, br, bl], dtype="float32")

        # --- TARGET TEMPLATE DIMENSIONS ---
        TARGET_W, TARGET_H = 1600, 2263

        # NOTE: If your final image looks squished or stretched,
        # you need to adjust these 4 coordinates to match your ideal layout!
        TL_TARGET = [496, 757]
        TR_TARGET = [1482, 757]
        BR_TARGET = [1482, 2146]
        BL_TARGET = [496, 2146]

        dst_pts = np.array([TL_TARGET, TR_TARGET, BR_TARGET, BL_TARGET], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        warped = cv2.warpPerspective(img, matrix, (TARGET_W, TARGET_H))

        return warped

    # --- IMAGE PROCESSING LOGIC ---
    def get_current_img(self):
        pil_img = self.pages[self.current_page_index]
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ALIGN THE IMAGE BEFORE THE GUI AND PROCESSING SEE IT
        aligned_img = self.align_image(img)
        return aligned_img

    def update(self, _=None):
        img = self.get_current_img()
        display_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.thresh_val.get(), 255, cv2.THRESH_BINARY_INV)

        self.process_logic(thresh, display_img)

        h, w = display_img.shape[:2]
        resized_display = cv2.resize(display_img, (w // 2, h // 2))
        cv2.imshow(f"Scanned Sheet - Page {self.current_page_index + 1}/{len(self.pages)}", resized_display)

    def process_logic(self, thresh_img, display_img):
        results = {}

        # --- HEADER PROCESSING ---
        h_y = self.header_y.get()
        h_rg = self.h_row_gap.get()
        h_cg = self.h_col_gap.get()
        bw = self.box_w.get()
        bh = self.box_h.get()

        # 1. Level (1-6)
        lvl_x = self.level_x.get()
        lvl_counts = []
        lvl_rects = []
        for r in range(6): # 6 levels from 1 to 6
            x1 = lvl_x
            y1 = h_y + r * (bh + h_rg)
            x2, y2 = x1 + bw, y1 + bh
            if y2 <= thresh_img.shape[0] and x2 <= thresh_img.shape[1]:
                lvl_counts.append(cv2.countNonZero(thresh_img[y1:y2, x1:x2]))
                lvl_rects.append((x1, y1, x2, y2))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if lvl_counts and max(lvl_counts) > 100:
            best = np.argmax(lvl_counts)
            results["Level"] = str(best + 1) # Levels are 1-6
            cv2.rectangle(display_img, lvl_rects[best][:2], lvl_rects[best][2:], (0, 255, 0), 2)
        else:
            results["Level"] = ""

        # 2. Class ('CH', 'FA', 'HO', 'JU')
        cls_x = self.class_x.get()
        class_options = ['CH', 'FA', 'HO', 'JU']
        cls_counts = []
        cls_rects = []
        for r, _ in enumerate(class_options):
            x1 = cls_x
            y1 = h_y + r * (bh + h_rg)
            x2, y2 = x1 + bw, y1 + bh
            if y2 <= thresh_img.shape[0] and x2 <= thresh_img.shape[1]:
                cls_counts.append(cv2.countNonZero(thresh_img[y1:y2, x1:x2]))
                cls_rects.append((x1, y1, x2, y2))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if cls_counts and max(cls_counts) > 100:
            best = np.argmax(cls_counts)
            results["Class"] = class_options[best]
            cv2.rectangle(display_img, cls_rects[best][:2], cls_rects[best][2:], (0, 255, 0), 2)
        else:
            results["Class"] = ""

        # 3. Class No (2 cols, 0-9)
        cno_x = self.c_no_x.get()
        cno_str = ""
        for c in range(2): # 2 columns for class number
            cx = cno_x + c * (bw + h_cg)
            col_counts = []
            col_rects = []
            for r in range(10):
                y1 = h_y + r * (bh + h_rg)
                x2, y2 = cx + bw, y1 + bh
                if y2 <= thresh_img.shape[0] and x2 <= thresh_img.shape[1]:
                    col_counts.append(cv2.countNonZero(thresh_img[y1:y2, cx:x2]))
                    col_rects.append((cx, y1, x2, y2))
                    cv2.rectangle(display_img, (cx, y1), (x2, y2), (0, 0, 255), 1)
            if col_counts and max(col_counts) > 100:
                best = np.argmax(col_counts)
                cno_str += str(best)
                cv2.rectangle(display_img, col_rects[best][:2], col_rects[best][2:], (0, 255, 0), 2)
            else:
                cno_str += "?"
        results["ClassNo"] = cno_str

        # 4. Category (A-H)
        cat_x = self.category_x.get()
        category_options = [chr(ord('A') + i) for i in range(8)] # A-H
        cat_counts = []
        cat_rects = []
        for r, _ in enumerate(category_options):
            x1 = cat_x
            y1 = h_y + r * (bh + h_rg)
            x2, y2 = x1 + bw, y1 + bh
            if y2 <= thresh_img.shape[0] and x2 <= thresh_img.shape[1]:
                cat_counts.append(cv2.countNonZero(thresh_img[y1:y2, x1:x2]))
                cat_rects.append((x1, y1, x2, y2))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if cat_counts and max(cat_counts) > 100:
            best = np.argmax(cat_counts)
            results["Category"] = category_options[best]
            cv2.rectangle(display_img, cat_rects[best][:2], cat_rects[best][2:], (0, 255, 0), 2)
        else:
            results["Category"] = ""

        # --- QUESTION PROCESSING ---
        options = ["A", "B", "C", "D"]
        total_questions = self.total_q.get()

        for q_num in range(1, total_questions + 1):
            q_idx = q_num - 1
            col_idx = q_idx // 25  # 0 for Q1-25, 1 for Q26-50, etc.
            q_in_col = q_idx % 25

            block = q_in_col // 5

            # Y position is the same for all columns
            base_y = self.start_y.get() + (q_in_col * (self.box_h.get() + self.q_gap.get())) + (block * self.block_gap.get())

            # X position depends on the column
            start_x = self.start_x.get()
            inner_col_gap = self.col_gap.get()
            main_col_gap = self.col_gap_main.get()

            if col_idx == 0: # Q1-25
                base_x = start_x
            elif col_idx == 1: # Q26-50
                base_x = start_x + inner_col_gap
            elif col_idx == 2: # Q51-75
                base_x = start_x + main_col_gap
            else: # Q76-100
                base_x = start_x + main_col_gap + inner_col_gap

            pixel_counts = []
            box_coords = []

            for opt_idx in range(4):
                x1 = base_x + (opt_idx * (self.box_w.get() + self.opt_gap.get()))
                y1 = base_y
                x2 = x1 + self.box_w.get()
                y2 = y1 + self.box_h.get()

                if x2 > thresh_img.shape[1] or y2 > thresh_img.shape[0]:
                    continue

                bubble = thresh_img[y1:y2, x1:x2]
                count = cv2.countNonZero(bubble)

                pixel_counts.append(count)
                box_coords.append((x1, y1, x2, y2))

                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Check how many bubbles are filled above the threshold
            filled_indices = [i for i, count in enumerate(pixel_counts) if count > 100]

            if len(filled_indices) == 1:
                # Exactly one bubble is filled - this is the correct case
                best_choice_idx = filled_indices[0]
                ans = options[best_choice_idx]
                bx1, by1, bx2, by2 = box_coords[best_choice_idx]
                # Draw a green box for the single valid answer
                cv2.rectangle(display_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            elif len(filled_indices) > 1:
                # More than one bubble is filled - mark as an error
                ans = "M"
                # Draw an orange box around all erroneously marked bubbles
                for idx in filled_indices:
                    bx1, by1, bx2, by2 = box_coords[idx]
                    cv2.rectangle(display_img, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
            else:
                # No bubbles are filled
                ans = ""

            results[q_num] = ans

        return results

    # --- PROGRESS & EXPORT LOGIC ---
    def next_page(self):
        # 1. Get the final answers for this page
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.thresh_val.get(), 255, cv2.THRESH_BINARY_INV)
        page_results = self.process_logic(thresh, img.copy())

        # 2. Add the page number so we know whose paper this is
        page_results["Page"] = self.current_page_index + 1
        self.all_results.append(page_results)
        print(f"Page {self.current_page_index + 1} confirmed and saved.")

        # 3. Move to next page or export if finished
        if self.current_page_index < len(self.pages) - 1:
            self.current_page_index += 1
            self.update()
        else:
            self.export_csv()
            print("Finished all pages! GUI closing.")
            self.root.destroy()
            cv2.destroyAllWindows()

    def export_csv(self):
        # Create CSV header: ["Page", 1, 2, 3... 50]
        if not self.all_results:
            print("No results to export.")
            return

        max_q = 0
        for res in self.all_results:
            q_keys = [k for k in res.keys() if isinstance(k, int)]
            if q_keys:
                max_q = max(max_q, max(q_keys))
        fieldnames = ["Page", "Level", "Class", "ClassNo", "Category"] + list(range(1, max_q + 1))
        filename = "student_scores.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.all_results:
                writer.writerow(row)

        print(f"SUCCESS: All results exported to {filename}")

# --- RUN THE APP ---
# Replace with your actual PDF file name
scanner = PDFOMRScanner("newtest.pdf")
