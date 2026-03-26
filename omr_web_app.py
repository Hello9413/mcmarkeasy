import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import json
import pandas as pd
import os
import io

# --- CONFIGURATION DEFAULTS ---
DEFAULT_CONFIG = {
    "thresh": 150, "total_q": 50,
    "start_x": 100, "start_y": 100, "box_w": 30, "box_h": 15,
    "opt_gap": 25, "q_gap": 20, "block_gap": 40, "col_gap": 500, "col_gap_main": 700,
    "header_y": 50, "level_x": 50, "class_x": 150, "c_no_x": 300, "category_x": 450,
    "h_row_gap": 5, "h_col_gap": 10
}

CONFIG_FILE = "omr_config.json"

# --- HELPER FUNCTIONS ---
def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                config.update(saved)
        except:
            pass
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    st.success("Configuration saved locally to omr_config.json")

def align_image(img):
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_markers = []
    # Updated logic for bigger square boxes
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
                            valid_markers.append((cx, cy))

    if len(valid_markers) != 4:
        return img, False # Failed alignment

    # Sort markers
    valid_markers = sorted(valid_markers, key=lambda x: x[1])
    top_markers = sorted(valid_markers[:2], key=lambda x: x[0])
    bottom_markers = sorted(valid_markers[-2:], key=lambda x: x[0])

    tl, tr = top_markers[0], top_markers[1]
    bl, br = bottom_markers[0], bottom_markers[1]

    ordered_pts = np.array([tl, tr, br, bl], dtype="float32")

    # Target dimensions
    TARGET_W, TARGET_H = 1600, 2263
    TL_TARGET = [496, 757]
    TR_TARGET = [1482, 757]
    BR_TARGET = [1482, 2146]
    BL_TARGET = [496, 2146]

    dst_pts = np.array([TL_TARGET, TR_TARGET, BR_TARGET, BL_TARGET], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (TARGET_W, TARGET_H))

    return warped, True

def process_page(img, cfg):
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, cfg['thresh'], 255, cv2.THRESH_BINARY_INV)

    results = {}

    # --- Helper to draw and count ---
    def check_region(x, y, w, h):
        if y+h > thresh_img.shape[0] or x+w > thresh_img.shape[1]: return 0
        return cv2.countNonZero(thresh_img[y:y+h, x:x+w])

    # --- Header Processing ---
    h_y, bw, bh = cfg['header_y'], cfg['box_w'], cfg['box_h']
    h_rg = cfg['h_row_gap']

    # 1. Level (1-6)
    lvl_counts = []
    for r in range(6):
        lx, ly = cfg['level_x'], h_y + r*(bh+h_rg)
        lvl_counts.append(check_region(lx, ly, bw, bh))
        cv2.rectangle(display_img, (lx, ly), (lx+bw, ly+bh), (0, 0, 255), 1)

    if max(lvl_counts) > 100:
        best = np.argmax(lvl_counts)
        results["Level"] = str(best + 1)
        y_pos = h_y + best*(bh+h_rg)
        cv2.rectangle(display_img, (cfg['level_x'], y_pos), (cfg['level_x']+bw, y_pos+bh), (0, 255, 0), 2)

    # 2. Class (CH, FA, HO, JU)
    cls_opts = ['CH', 'FA', 'HO', 'JU']
    cls_counts = []
    for r in range(4):
        cx, cy = cfg['class_x'], h_y + r*(bh+h_rg)
        cls_counts.append(check_region(cx, cy, bw, bh))
        cv2.rectangle(display_img, (cx, cy), (cx+bw, cy+bh), (0, 0, 255), 1)

    if max(cls_counts) > 100:
        best = np.argmax(cls_counts)
        results["Class"] = cls_opts[best]
        y_pos = h_y + best*(bh+h_rg)
        cv2.rectangle(display_img, (cfg['class_x'], y_pos), (cfg['class_x']+bw, y_pos+bh), (0, 255, 0), 2)

    # 3. Class No (2 cols, 0-9)
    cno_x = cfg['c_no_x']
    h_cg = cfg['h_col_gap']
    cno_str = ""
    for c_idx in range(2):
        col_x = cno_x + c_idx * (bw + h_cg)
        col_counts = []
        for r in range(10):
            ry = h_y + r*(bh+h_rg)
            col_counts.append(check_region(col_x, ry, bw, bh))
            cv2.rectangle(display_img, (col_x, ry), (col_x+bw, ry+bh), (0, 0, 255), 1)

        if max(col_counts) > 100:
            best = np.argmax(col_counts)
            cno_str += str(best)
            y_pos = h_y + best*(bh+h_rg)
            cv2.rectangle(display_img, (col_x, y_pos), (col_x+bw, y_pos+bh), (0, 255, 0), 2)
        else:
            cno_str += "?"
    results["ClassNo"] = cno_str

    # 4. Category (A-H)
    cat_x = cfg['category_x']
    cat_opts = [chr(ord('A') + i) for i in range(8)]
    cat_counts = []
    for r in range(8):
        cy = h_y + r*(bh+h_rg)
        cat_counts.append(check_region(cat_x, cy, bw, bh))
        cv2.rectangle(display_img, (cat_x, cy), (cat_x+bw, cy+bh), (0, 0, 255), 1)

    if max(cat_counts) > 100:
        best = np.argmax(cat_counts)
        results["Category"] = cat_opts[best]
        y_pos = h_y + best*(bh+h_rg)
        cv2.rectangle(display_img, (cat_x, y_pos), (cat_x+bw, y_pos+bh), (0, 255, 0), 2)

    # --- Question Processing ---
    options = ["A", "B", "C", "D"]
    for q_num in range(1, cfg['total_q'] + 1):
        q_idx = q_num - 1
        col_idx = q_idx // 25
        q_in_col = q_idx % 25
        block = q_in_col // 5

        base_y = cfg['start_y'] + (q_in_col * (cfg['box_h'] + cfg['q_gap'])) + (block * cfg['block_gap'])

        if col_idx == 0: base_x = cfg['start_x']
        elif col_idx == 1: base_x = cfg['start_x'] + cfg['col_gap']
        elif col_idx == 2: base_x = cfg['start_x'] + cfg['col_gap_main']
        else: base_x = cfg['start_x'] + cfg['col_gap_main'] + cfg['col_gap']

        counts = []
        coords = []
        for i in range(4):
            x = base_x + (i * (cfg['box_w'] + cfg['opt_gap']))
            counts.append(check_region(x, base_y, cfg['box_w'], cfg['box_h']))
            coords.append((x, base_y))
            cv2.rectangle(display_img, (x, base_y), (x+cfg['box_w'], base_y+cfg['box_h']), (0, 0, 255), 1)

        filled = [i for i, c in enumerate(counts) if c > 100]
        if len(filled) == 1:
            ans = options[filled[0]]
            x, y = coords[filled[0]]
            cv2.rectangle(display_img, (x, y), (x+cfg['box_w'], y+cfg['box_h']), (0, 255, 0), 3)
        elif len(filled) > 1:
            ans = "M" # Multiple
            # Highlight all erroneously marked bubbles in Orange (BGR: 0, 165, 255)
            for idx in filled:
                x, y = coords[idx]
                cv2.rectangle(display_img, (x, y), (x+cfg['box_w'], y+cfg['box_h']), (0, 165, 255), 3)
        else:
            ans = ""

        results[q_num] = ans

    return display_img, results

# --- MAIN APP ---
st.set_page_config(layout="wide", page_title="OMR Web App")
st.title("OMR Scanner & Configurator")

# Session State
if 'page_index' not in st.session_state: st.session_state.page_index = 0
if 'images' not in st.session_state: st.session_state.images = []
if 'results_df' not in st.session_state: st.session_state.results_df = pd.DataFrame()
if 'model_answer' not in st.session_state: st.session_state.model_answer = None
if 'batch_active' not in st.session_state: st.session_state.batch_active = False
if 'batch_index' not in st.session_state: st.session_state.batch_index = -1
if 'zoom' not in st.session_state: st.session_state.zoom = 0.5  # Start at 50% scale

# Sidebar Controls
st.sidebar.header("Settings")
config = load_config()
new_config = {}

with st.sidebar.expander("1. Detection Settings", expanded=True):
    new_config['thresh'] = st.slider("Threshold", 0, 255, config['thresh'])
    new_config['total_q'] = st.slider("Total Questions", 1, 100, config['total_q'])

with st.sidebar.expander("2. Position & Size", expanded=False):
    new_config['start_x'] = st.slider("Start X", 0, 800, config['start_x'])
    new_config['start_y'] = st.slider("Start Y", 0, 900, config['start_y'])
    new_config['box_w'] = st.slider("Box Width", 10, 100, config['box_w'])
    new_config['box_h'] = st.slider("Box Height", 10, 100, config['box_h'])

with st.sidebar.expander("3. Spacing", expanded=False):
    new_config['opt_gap'] = st.slider("Option Gap", 0, 100, config['opt_gap'])
    new_config['q_gap'] = st.slider("Question Gap", 0, 100, config['q_gap'])
    new_config['col_gap'] = st.slider("Col Gap", 200, 1200, config['col_gap'])
    new_config['col_gap_main'] = st.slider("Main Gap", 200, 1200, config['col_gap_main'])
    new_config['block_gap'] = st.slider("Block Gap", 0, 200, config['block_gap'])

with st.sidebar.expander("4. Header Settings", expanded=False):
    new_config['header_y'] = st.slider("Header Y", 0, 600, config['header_y'])
    new_config['level_x'] = st.slider("Level X", 0, 1200, config['level_x'])
    new_config['class_x'] = st.slider("Class X", 0, 1400, config['class_x'])
    new_config['c_no_x'] = st.slider("Class No X", 0, 1600, config['c_no_x'])
    new_config['category_x'] = st.slider("Category X", 0, 1600, config['category_x'])
    new_config['h_row_gap'] = st.slider("Header Row Gap", 0, 50, config['h_row_gap'])
    new_config['h_col_gap'] = st.slider("Header Col Gap", 0, 50, config['h_col_gap'])

# Pass remaining configs that aren't sliders yet
for k, v in config.items():
    if k not in new_config:
        new_config[k] = v

if st.sidebar.button("Save Configuration"):
    save_config(new_config)

# File Upload
uploaded_file = st.file_uploader("Upload Exam PDF", type=['pdf'])

if uploaded_file:
    if not st.session_state.images:
        with st.spinner("Processing PDF..."):
            images = convert_from_bytes(uploaded_file.read(), dpi=200)
            st.session_state.images = images
            st.session_state.page_index = 0
            # Reset results
            st.session_state.results_df = pd.DataFrame(columns=["Page", "Level", "Class", "ClassNo", "Category", "Score"] + list(range(1, new_config['total_q']+1)))

    if st.session_state.images:
        # Navigation and Zoom Controls
        nav_cols = st.columns([1, 1, 1, 1, 1])
        if nav_cols[0].button("⬅️ Previous") and st.session_state.page_index > 0:
            st.session_state.page_index -= 1
        if nav_cols[1].button("Next ➡️") and st.session_state.page_index < len(st.session_state.images) - 1:
            st.session_state.page_index += 1
        if nav_cols[2].button("🔍 Zoom Out"):
            st.session_state.zoom = max(0.1, st.session_state.zoom - 0.1)
        if nav_cols[3].button("🔄 Reset Zoom"):
            st.session_state.zoom = 0.5
        if nav_cols[4].button("🔍 Zoom In"):
            st.session_state.zoom = min(2.0, st.session_state.zoom + 0.1)

        curr_idx = st.session_state.page_index
        pil_img = st.session_state.images[curr_idx]
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Align
        warped, success = align_image(cv_img)

        if not success:
            st.error("Could not find 4 markers! Showing raw image.")
            st.image(cv_img, channels="BGR")
        else:
            # Process with live sliders
            processed_img, page_results = process_page(warped, new_config)

            # Display with dynamic width based on zoom
            st.image(
                processed_img,
                channels="BGR",
                width=int(1600 * st.session_state.zoom),
                caption=f"Page {curr_idx + 1} (Zoom: {int(st.session_state.zoom*100)}%)"
            )

            # Grading and Saving Logic
            st.divider()
            col_m1, col_m2 = st.columns(2)

            if col_m1.button("🏆 Save as Model Answer"):
                st.session_state.model_answer = page_results
                st.session_state.batch_active = True
                st.session_state.batch_index = curr_idx + 1
                st.success("Model answer set. Starting automated batch processing...")
                st.rerun()

            if col_m2.button("📝 Save Student Result"):
                score = 0
                if st.session_state.model_answer is None:
                    st.warning("Please set a Model Answer first to calculate scores.")
                else:
                    # Compare student answers to model answer
                    for q in range(1, new_config['total_q'] + 1):
                        student_ans = page_results.get(q, "")
                        model_ans = st.session_state.model_answer.get(q, "")
                        # Only count as correct if it's not empty and matches exactly
                        if student_ans != "" and student_ans == model_ans:
                            score += 1

                    # Prepare the row data
                    row_data = {
                        "Page": curr_idx + 1,
                        "Level": page_results.get("Level"),
                        "Class": page_results.get("Class"),
                        "ClassNo": page_results.get("ClassNo"),
                        "Category": page_results.get("Category"),
                        "Score": score
                    }
                    # Add individual question answers
                    for q in range(1, new_config['total_q'] + 1):
                        row_data[q] = page_results.get(q)

                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row_data])], ignore_index=True)
                    st.success(f"Student Result Saved! Score: {score} / {new_config['total_q']}")

            if st.session_state.model_answer:
                st.info("✅ Model Answer is loaded.")

        # --- Batch Processing & Manual Intervention Logic ---
        if st.session_state.batch_active:
            b_idx = st.session_state.batch_index
            if b_idx < len(st.session_state.images):
                st.divider()
                st.header(f"⚡ Batch Processing: Page {b_idx + 1}")

                # Process current batch page
                b_pil = st.session_state.images[b_idx]
                b_cv = cv2.cvtColor(np.array(b_pil), cv2.COLOR_RGB2BGR)
                b_warped, b_success = align_image(b_cv)

                if not b_success:
                    st.error(f"Page {b_idx+1} failed alignment. Skipping.")
                    st.session_state.batch_index += 1
                    st.rerun()

                b_processed, b_results = process_page(b_warped, new_config)

                # Identify issues (Blank or Multi)
                issues = [q for q, a in b_results.items() if isinstance(q, int) and a in ["", "M"]]

                if not issues:
                    # Auto-save clean sheet
                    score_b = sum(1 for q in range(1, new_config['total_q']+1)
                                 if b_results.get(q) != "" and b_results.get(q) == st.session_state.model_answer.get(q))

                    row_b = {
                        "Page": b_idx + 1,
                        "Level": b_results.get("Level"), "Class": b_results.get("Class"),
                        "ClassNo": b_results.get("ClassNo"), "Category": b_results.get("Category"),
                        "Score": score_b
                    }
                    for q in range(1, new_config['total_q'] + 1): row_b[q] = b_results.get(q)

                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row_b])], ignore_index=True)
                    st.session_state.batch_index += 1
                    st.rerun()
                else:
                    # Show Manual Fix UI
                    st.warning(f"Manual intervention required for Page {b_idx+1}. Questions with issues: {issues}")
                    st.image(b_processed, channels="BGR", width=int(1600 * st.session_state.zoom))

                    with st.form(f"fix_form_{b_idx}"):
                        st.subheader("Correct the answers below:")
                        manual_fixes = {}
                        cols_fix = st.columns(min(len(issues), 5))
                        for i, q_num in enumerate(issues):
                            with cols_fix[i % 5]:
                                current_val = b_results.get(q_num)
                                label = f"Q{q_num} (Detected: '{current_val}')"
                                manual_fixes[q_num] = st.selectbox(label, ["Blank", "A", "B", "C", "D", "M"],
                                                                 index=0 if current_val == "" else 5)

                        if st.form_submit_button("✅ Confirm Fixes & Continue"):
                            # Apply fixes
                            for q, val in manual_fixes.items():
                                b_results[q] = "" if val == "Blank" else val

                            # Recalculate score
                            score_b = sum(1 for q in range(1, new_config['total_q']+1)
                                         if b_results.get(q) != "" and b_results.get(q) == st.session_state.model_answer.get(q))

                            row_b = {
                                "Page": b_idx + 1,
                                "Level": b_results.get("Level"), "Class": b_results.get("Class"),
                                "ClassNo": b_results.get("ClassNo"), "Category": b_results.get("Category"),
                                "Score": score_b
                            }
                            for q in range(1, new_config['total_q'] + 1): row_b[q] = b_results.get(q)

                            st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row_b])], ignore_index=True)
                            st.session_state.batch_index += 1
                            st.rerun()

                    if st.button("⏹️ Stop Batch"):
                        st.session_state.batch_active = False
                        st.rerun()
            else:
                st.session_state.batch_active = False
                st.success("Batch Processing Complete!")

    if not st.session_state.results_df.empty:
        st.divider()
        if st.session_state.model_answer:
            st.subheader("Model Answer (Answer Key)")
            # Prepare model answer row with the same columns as student results
            model_row = {
                "Page": "KEY",
                "Level": st.session_state.model_answer.get("Level"),
                "Class": st.session_state.model_answer.get("Class"),
                "ClassNo": st.session_state.model_answer.get("ClassNo"),
                "Category": st.session_state.model_answer.get("Category"),
                "Score": "-"
            }
            for q in range(1, new_config['total_q'] + 1):
                model_row[q] = st.session_state.model_answer.get(q)

            cols = ["Page", "Level", "Class", "ClassNo", "Category", "Score"] + list(range(1, new_config['total_q'] + 1))
            st.dataframe(pd.DataFrame([model_row], columns=cols))
            st.divider()
        st.subheader("Final Results")
        st.dataframe(st.session_state.results_df)

        # Create Multi-Sheet Excel Report
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            # Define formatting styles
            center_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
            pct_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '0.0"%"'})

            # 1. Student info with score only
            summary_df = st.session_state.results_df[["Level", "Class", "ClassNo", "Category", "Score"]].copy()
            summary_df["Percentage"] = (summary_df["Score"] / new_config['total_q']) * 100
            summary_df.to_excel(writer, index=False, sheet_name='Student Summary')
            writer.sheets['Student Summary'].set_column(0, 4, 12, center_fmt)
            writer.sheets['Student Summary'].set_column(5, 5, 15, pct_fmt)

            # 2. Statistics for each question
            stats_list = []
            total_students = len(st.session_state.results_df)
            for q in range(1, new_config['total_q'] + 1):
                col_data = st.session_state.results_df[q]
                model_ans = st.session_state.model_answer.get(q) if st.session_state.model_answer else None
                correct_count = (col_data == model_ans).sum() if model_ans else 0

                stats_list.append({
                    "Question": q,
                    "Correct Answer": model_ans,
                    "A": (col_data == "A").sum(),
                    "B": (col_data == "B").sum(),
                    "C": (col_data == "C").sum(),
                    "D": (col_data == "D").sum(),
                    "Multi Answer": (col_data == "M").sum(),
                    "Blank": (col_data == "").sum(),
                    "Percentage Correct": (correct_count / total_students * 100) if total_students > 0 else 0
                })
            stats_df = pd.DataFrame(stats_list)
            stats_df.to_excel(writer, index=False, sheet_name='Question Statistics')
            writer.sheets['Question Statistics'].set_column(0, 7, 14, center_fmt)
            writer.sheets['Question Statistics'].set_column(8, 8, 18, pct_fmt)

            # 3. Final Results (No Page column)
            # This drops the 'Page' column and everything that might have been before it
            final_report = st.session_state.results_df.drop(columns=["Page"])
            final_report.to_excel(writer, index=False, sheet_name='Full Details')
            writer.sheets['Full Details'].set_column(0, len(final_report.columns) - 1, 10, center_fmt)

        st.download_button(
            label="📊 Download Multi-Sheet Excel Report",
            data=output.getvalue(),
            file_name="omr_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
