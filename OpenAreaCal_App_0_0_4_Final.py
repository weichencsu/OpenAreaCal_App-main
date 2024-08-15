# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from streamlit_option_menu import option_menu
import av
import cv2
import time
import threading
from typing import Union
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os


def create_polygon_mask(image_shape, vertices_list):
    # 创建一个与图片大小相同的黑色图像
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # 在mask上绘制所有的多边形
    for vertices in vertices_list:
        cv2.fillPoly(mask, [vertices], 255)

    # 将mask扩展到3个通道
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def mask_polygons(image, vertices_list):
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 创建包含所有多边形的mask
    mask = create_polygon_mask(image_cv.shape, vertices_list)

    # 保留多边形内的像素，外部变为白色
    masked_image = cv2.bitwise_and(image_cv, mask)
    white_background = np.ones_like(image_cv) * 255
    masked_image = cv2.bitwise_or(masked_image, cv2.bitwise_and(white_background, cv2.bitwise_not(mask)))

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))


def mask_polygon_inverse(image, vertices):
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Create a black mask
    mask = np.zeros_like(image_cv)
    # Create a white polygon
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    # Keep pixels outside the polygon, make inside black
    masked_image = cv2.bitwise_and(image_cv, inverted_mask)

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))


def extract_coordinates(path):
    coordinates = []
    for point in path:
        if point[0] in ['M', 'L']:  # M is start point, L is line point
            coordinates.append([int(point[1]), int(point[2])])
    return np.array(coordinates, dtype=np.int32)


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    # Load the mask image and resize it to match the camera input
    mask = cv2.imread('mask.jpg')
    mask = cv2.resize(mask, (640, 480))

    # Convert the mask to a binary mask using thresholding
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary_mask = cv2.threshold(gray_mask, 250, 255, cv2.THRESH_BINARY)

    # Convert the binary mask to 3 channels to match the camera input
    color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, color_mask)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def crop_and_resize(image, target_width=396, target_height=649):  # 标准尺寸应为 300 480
    # 检查输入是否为 PIL Image 对象
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        # 假设输入是 numpy 数组，转换为 PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    # # 将图像从 BGR 转换为 RGB
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    # # 创建 PIL Image 对象
    # pil_image = Image.fromarray(image_rgb)

    # 获取原始图像尺寸
    width, height = pil_image.size
    # 计算裁剪区域
    aspect_ratio = target_width / target_height
    if width / height > aspect_ratio:
        # 图像太宽，需要裁剪宽度
        new_width = int(height * aspect_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height
    else:
        # 图像太高，需要裁剪高度
        new_height = int(width / aspect_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width

    # 裁剪图像
    cropped_image = pil_image.crop((left, top, right, bottom))

    # 调整图像大小
    resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

    # 转换回 numpy 数组并返回 BGR 格式
    return cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)


def normalise_data_right(df):
    # 确保列名正确
    manual_col = "Slot Size - mm [Manual]"
    ai_area_col = "Slot Area - Pixels"
    ai_size_col = "Slot Size - Pixels"

    # 创建一个新的列，存储转换后的手动测量值，同时保留原始的空字符串
    df["manual_numeric"] = pd.to_numeric(df[manual_col].replace(' ', None), errors='coerce')

    # 找出有效的行（非NaN的数值）
    valid_rows = df["manual_numeric"].notna()

    if valid_rows.any():
        # 计算比率，只使用有效行
        mean_ratio = (df.loc[valid_rows, ai_size_col] / df.loc[valid_rows, "manual_numeric"]).mean()

        # 更新DataFrame，使用fillna来处理可能的除以零的情况
        df["Slot Size - mm [Calculated]"] = (df[ai_size_col] / mean_ratio).round(1).fillna(df[ai_area_col])
        df["Slot Area - mm2 [Calculated]"] = (df[ai_area_col] / mean_ratio * 3.92).round(1).fillna(df[ai_area_col])
        total_area = df["Slot Area - mm2 [Calculated]"].sum()
        # st.write(f"Calibration ratio: {mean_ratio:.4f}")
    else:
        # st.write("No valid manual measurements found for calibration. Using original AI data.")
        mean_ratio = 1
        df["Slot Size - mm [Calculated]"] = df[ai_size_col]
        df["Slot Area - mm2 [Calculated]"] = df[ai_area_col]
        total_area = df["Slot Area - mm2 [Calculated]"].sum()

    # 删除辅助列并保留原始的空字符串
    df.drop(columns=["manual_numeric"], inplace=True)

    return df, total_area, mean_ratio


def watershed_segmentation(img, ori_input_img, identification_thresh=76):
    # 添加面积计算与导出
    # # 读取图像
    img = np.array(img)
    # 图像处理
    shifted = cv2.pyrMeanShiftFiltering(img, 10, 25)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, identification_thresh, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    sure_bg = cv2.dilate(opening, kernel, iterations=4)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    alpha = 0.01
    ret, sure_fg = cv2.threshold(dist_transform, alpha * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 准备显示图像
    # img1 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(ori_input_img.copy(), cv2.COLOR_BGR2RGB)
    img1[markers == -1] = [255, 0, 0]  # 分水岭标蓝色

    # # 转换为PIL图像以便Streamlit显示
    watershed_img = Image.fromarray(img1)

    # 获取唯一的标签（跳过背景，通常是0）
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != 1) & (unique_labels != -1)]

    # 为每个孔创建掩码并找到中心
    valid_hole_count = 0
    left_total_area = []
    right_total_area = []
    hole_id_list = []
    hole_areas_list = []
    vertical_lengths_list = []
    img_center_x = ori_input_img.shape[1] // 2
    for label in unique_labels:
        # 创建二值掩码
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255

        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 对于每个轮廓（通常只有一个）
        for contour in contours:
            pixel_count = cv2.countNonZero(mask)

            # 只处理面积大于等于min_area的区域
            if pixel_count >= 300:
                # 计算轮廓的矩
                M = cv2.moments(contour)
                # # 计算面积
                # area = M["m00"]
                valid_hole_count += 1
                # 计算中心点
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"] - 15)
                    cY = int(M["m01"] / M["m00"] + 10)
                    # 在原始图像上绘制标签
                    cv2.putText(ori_input_img, f"#{valid_hole_count}", (cX, cY),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 0), 1)
                    # 绘制轮廓
                    cv2.drawContours(ori_input_img, [contour], -1, (0, 255, 0), 1)

                    # 计算边界矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    vertical_lengths_list.append(h)  # 添加竖直方向长度
                    hole_id_list.append(valid_hole_count)
                    hole_areas_list.append(pixel_count)
                    print(f"Hole #{valid_hole_count} area: {pixel_count} vertical length: {h}")
                    # 判断孔洞是在左侧还是右侧,并加到对应的总像素点中
                    if cX < img_center_x:
                        left_total_area.append(pixel_count)
                    else:
                        right_total_area.append(pixel_count)

    if len(hole_areas_list) != 0:
        hole_areas = hole_areas_list
    else:
        hole_areas = None

    if len(vertical_lengths_list) != 0:
        vertical_lengths = vertical_lengths_list
    else:
        vertical_lengths = None

    return ori_input_img, hole_id_list, hole_areas, vertical_lengths


def black_to_white_mask(image):
    # 转换图像到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黑色的 HSV 范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # 创建黑色区域的掩码
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 创建白色图像
    white_image = np.ones_like(image) * 255

    # 将黑色区域替换为白色
    result = np.where(mask[:, :, None].astype(bool), white_image, image)

    return result


def get_total_days(file_path):
    df = pd.read_excel(file_path, sheet_name='Service details')
    total_days = df['Total Days Since Reline'].iloc[0]
    total_area = df['Total Areas'].iloc[0]
    total_tons = df['Total Tons Milled'].iloc[0]

    return total_days, total_area, total_tons


def main():
    st.set_page_config(page_title='BHP ODO Svedala Grates Open Area Calculation Tool', initial_sidebar_state='auto')
    st.markdown(
        f"""
            <style>
                .reportview-container .main .block-container{{
                    max-width: 1400px;
                    padding-top: 1rem;
                    padding-right: 1rem;
                    padding-left: 1rem;
                    padding-bottom: 1rem;
                }}

            </style>
            """,
        unsafe_allow_html=True,
    )
    styles = {
        # "container": {"margin": "0px !important", "padding": "0!important", "align-items": "stretch",
        #               "background-color": "#fafafa", "font-size": "13px"},
        "icon": {"color": "black", "font-size": "14px"},
        "nav-link": {"font-size": "13px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "15px", "font-weight": "normal", },
    }
    PAGES = {
        "Grate Recognition": recognition_app,
        "Data Analysis": analysis_app
    }
    with st.sidebar:
        selected = option_menu("Bradken", ["Grate Recognition", "Data Analysis",],
                               icons=['camera', 'bar-chart'],  menu_icon='house', default_index=0, styles=styles)  # menu_icon="cast",
    PAGES[selected]()


def recognition_app():
    class VideoProcessor(VideoProcessorBase):
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            in_image = frame.to_ndarray(format="bgr24")
            in_image = cv2.flip(in_image, 1)
            # 这里是您的图像处理代码
            mask = cv2.imread('mask.jpg')
            mask = cv2.resize(mask, (640, 480))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, binary_mask = cv2.threshold(gray_mask, 250, 255, cv2.THRESH_BINARY)
            color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            out_image = cv2.bitwise_and(in_image, color_mask)

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            # 将处理后的图像转换回 VideoFrame
            return av.VideoFrame.from_ndarray(out_image, format="bgr24")


    titleContainer = st.container()
    with titleContainer:
        titleColmns1, titleColmns2 = st.columns([3.2, 1.0], gap='small')
        with titleColmns1:
            st.title("Bradken - BHP ODO")
            st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                        ' color:Black; font-size: 24px; ">Svedala Discharge End Open Area AI </p></nobr>',
                        unsafe_allow_html=True)
        with titleColmns2:
            st.image('bradken.png')
            # st_lottie(lottie_coding, height=150, key="hello")

    st.markdown("-------------------")

    hole_layout = st.selectbox(
        'Please select which grate to analyse',
        ('22mm Grate', '65mm Pebble Grate', '22mm middle Grate'))

    st.markdown('---')

    if 'snapshot_clicked' not in st.session_state:
        st.session_state.snapshot_clicked = False
    if 'submitted_image' not in st.session_state:
        st.session_state.submitted_image = None
    if 'submitted_image2' not in st.session_state:
        st.session_state.submitted_image2 = None
    if 'snapshot_image' not in st.session_state:
        st.session_state.snapshot_image = None

    videoContainer = st.container()
    with videoContainer:
        videoColmns1, videoColmns2 = st.columns([1, 1], gap='small')
        with videoColmns1:
            st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                        ' color:Black; font-size: 15px; ">Please capture the image of the selected grate</p></nobr>',
                        unsafe_allow_html=True)
            ctx = webrtc_streamer(key="snapshot", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={
                                      "video": True,
                                      "audio": False, },
                                  rtc_configuration={  # Add this line
                                      "iceServers": [{"urls": ["stun:stun1.l.google.com:19302"
                                                               ]}]  # "stun.flashdance.cx:3478"
                                      # "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                                  }
                                  )
        with videoColmns2:
            st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                        ' color:Black; font-size: 15px; margin-bottom: 24px;">Snapshot display</p></nobr>',
                        unsafe_allow_html=True)


            if ctx.video_processor:
                # 创建一个空白的占位符
                placeholder = st.empty()

                # 如果snapshot没有被点击，显示灰色的空白区域
                if not st.session_state.snapshot_clicked:
                    placeholder.markdown(
                        """
                        <div style="background-color: #f0f0f0; height: 258px; width: 100%; margin-bottom: 25px; "></div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # 如果snapshot被点击，显示snapshot_image
                    if st.session_state.snapshot_image is not None:
                        placeholder.image(st.session_state.snapshot_image, channels="BGR")
                    else:
                        placeholder.markdown(
                            """
                            <div style="background-color: #f0f0f0; height: 258px; width: 100%; margin-top: 0px; 
                            margin-bottom: 7px;"></div>
                            """,
                            unsafe_allow_html=True
                        )

                # 创建两列布局
                buttonscol1, buttonscol2 = st.columns([1, 1])

                # SNAPSHOT按钮
                if buttonscol1.button("SNAPSHOT"):
                    st.session_state.snapshot_clicked = True
                    with ctx.video_processor.frame_lock:
                        st.session_state.snapshot_image = ctx.video_processor.out_image
                    st.experimental_rerun()

                # Submit按钮
                if buttonscol2.button("SELECT THE IMAGE"):
                    if st.session_state.snapshot_clicked and st.session_state.snapshot_image is not None:
                        # st.session_state.submitted_image = st.session_state.snapshot_image
                        processed_image = black_to_white_mask(crop_and_resize(st.session_state.snapshot_image))
                        st.session_state.submitted_image = processed_image
                        st.session_state.submitted_image2 = black_to_white_mask(st.session_state.snapshot_image)

                    st.experimental_rerun()

            else:
                # 创建一个空白的占位符
                placeholder = st.empty()

                # 显示灰色的空白区域
                placeholder.markdown(
                    """
                    <div style="background-color: #F0F2F6; height: 100px; width: 100%; margin-top: 0px;"></div>
                    """,
                    unsafe_allow_html=True
                )
    st.image(Image.fromarray(st.session_state.submitted_image))
    st.image(Image.fromarray(st.session_state.submitted_image2))
    
    st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                ' color:Black; font-size: 15px; ">Or select an existing grates image</p></nobr>',
                unsafe_allow_html=True)
    bg_image = st.file_uploader("Upload image:", label_visibility='collapsed', type=["png", "jpg"])  # , type=["csv", "txt"]

    st.markdown('---')
    st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                ' color:Black; font-size: 15px; ">Select the recognition area</p></nobr>', unsafe_allow_html=True)

    img_preprocess_container = st.container()
    with img_preprocess_container:
        if 'recognition_clicked' not in st.session_state:
            st.session_state.recognition_clicked = False
        if 'recognition_view_image' not in st.session_state:
            st.session_state.recognition_view_image = Image.new("RGBA", (649, 396), "#eee")
        if 'recognition_image' not in st.session_state:
            st.session_state.recognition_image = None

        bg_color = "#eee"
        # bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
        # realtime_update = st.checkbox("Update in realtime", True)

        if st.session_state.submitted_image2 is not None:
            bg_image_canvas1 = Image.fromarray(cv2.cvtColor(st.session_state.submitted_image2, cv2.COLOR_BGR2RGB)).resize((649, 396))
        elif bg_image:
            bg_image_canvas1 = Image.open(bg_image).resize((649, 396))
        else:
            bg_image_canvas1 = Image.new("RGBA", (649, 396), "#eee")

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,
            # stroke_color=stroke_color,
            background_color=bg_color,
            background_image=bg_image_canvas1 if bg_image_canvas1 is not None else None,
            update_streamlit=True,
            height=396,
            width=649,
            drawing_mode="polygon",
            point_display_radius=0,
            display_toolbar=True,
            key="full_app",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            result_image = Image.new("RGBA", (649, 396), bg_color)

            if st.session_state.submitted_image2 is not None or bg_image:
                background = bg_image_canvas1
                result_image.paste(background, (0, 0))
                result_image2 = background
            else:
                result_image2 = result_image

            # Paste the drawn content onto the background
            draw_image = Image.fromarray(canvas_result.image_data)
            result_image.paste(draw_image, (0, 0), draw_image)

            # Process polygons
            if canvas_result.json_data is not None:
                vertices_list = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "path":
                        vertices = extract_coordinates(obj["path"])
                        vertices_list.append(vertices)

                if vertices_list:
                    result_image2 = mask_polygons(result_image2, vertices_list)
                st.image(result_image2)
        else:
            st.image(bg_image_canvas1)
        #     if st.session_state.recognition_view_image is not None:
        #         st.image(st.session_state.recognition_view_image)


        # Recognition/Crop 按钮
        if st.button("Confirm Cropped Image"):
            if st.session_state.submitted_image2 is not None or bg_image:
                st.session_state.recognition_clicked = True
                st.session_state.recognition_image = result_image2
            st.experimental_rerun()
        # if st.session_state.recognition_image is not None:
        #     st.image(st.session_state.recognition_image)
        # else:
        #     st.write("无Cropped图像")

    st.markdown('---')
    st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                ' color:Black; font-size: 15px; ">Cover image noise</p></nobr>', unsafe_allow_html=True)
    img_cover_noise_container = st.container()
    with img_cover_noise_container:
        if 'cover_noise_clicked' not in st.session_state:
            st.session_state.cover_noise_clicked = False
        # if 'cover_noise_view_image' not in st.session_state:
        #     st.session_state.cover_noise_view_image = Image.new("RGBA", (649, 396), "#eee")
        if 'cover_noise_image' not in st.session_state:
            st.session_state.cover_noise_image = None

        if st.session_state.recognition_image is not None:
            bg_image_canvas2 = st.session_state.recognition_image.resize((649, 396))
        else:
            bg_image_canvas2 = Image.new("RGBA", (649, 396), "#eee")

        # Create a canvas component
        canvas_result_inverse = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,
            # stroke_color=stroke_color,
            background_color=bg_color,
            background_image=st.session_state.recognition_image if st.session_state.recognition_image is not None else None,
            update_streamlit=True,
            height=396,
            width=649,
            drawing_mode="polygon",
            point_display_radius=0,
            display_toolbar=True,
            key="full_app_inverse",
        )

        if canvas_result_inverse.image_data is not None:
            result_image = Image.new("RGBA", (649, 396), bg_color)
            if st.session_state.recognition_image is not None:
                background = bg_image_canvas2
                result_image.paste(background, (0, 0))

            # Paste the drawn content onto the background
            draw_image = Image.fromarray(canvas_result_inverse.image_data)
            result_image.paste(draw_image, (0, 0), draw_image)

            # Process polygons
            if canvas_result_inverse.json_data is not None:
                # st.write(canvas_result.json_data)
                for obj in canvas_result_inverse.json_data["objects"]:
                    # st.write(extract_coordinates(obj["path"]))
                    if obj["type"] == "path":
                        vertices = extract_coordinates(obj["path"])
                        # You can choose either mask_polygon or mask_polygon_inverse here
                        # result_image = mask_polygon(result_image, vertices)
                        # Uncomment the line below and comment the line above to use inverse masking
                        result_image = mask_polygon_inverse(result_image, vertices)
                        # result_image = result_image

            st.image(result_image)
        else:
            st.image(bg_image_canvas2)

        # Recognition/Crop 按钮
        if st.button("Confirm Noise Covered Image"):
            if st.session_state.cover_noise_clicked is not None:
                st.session_state.cover_noise_clicked = True
                st.session_state.cover_noise_image = result_image
            st.experimental_rerun()

    st.markdown('---')
    st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                ' color:Black; font-size: 15px; ">Grates image</p></nobr>', unsafe_allow_html=True)

    r1col1, r1col2 = st.columns(2)
    with r1col1:
        r2col1s = r1col1.columns([1, 3, 1])
        # 显示确认的图像
        if st.session_state.cover_noise_image is not None:
            r2col1s[1].image(crop_and_resize(st.session_state.cover_noise_image), channels="BGR",)

        else:
            r2col1s[1].image("impage1.png", use_column_width=True, output_format="PNG")  # caption="Captured"
        st.markdown('<nobr><p style="text-align:center;font-family:sans serif;'
                    ' color:Black; font-size: 13px; ">Captured grate image</p></nobr>', unsafe_allow_html=True)

        identification_ratio = st.slider(
            "Please adjust identification ratio", min_value=1, max_value=255, value=76, step=1,
            label_visibility='collapsed',
        )

        if 'confirm_button' not in st.session_state:
            st.session_state.confirm_button = False
        if 'confirmed_image' not in st.session_state:
            st.session_state.confirmed_image = None
        if 'hole_id' not in st.session_state:
            st.session_state.hole_id = None
        if 'hole_areas' not in st.session_state:
            st.session_state.hole_areas = None
        if 'vertical_lengths' not in st.session_state:
            st.session_state.vertical_lengths = None
        # analyse_button = st.button("Analyse")
        if 'df_csv' not in st.session_state:
            st.session_state.df_csv = None  # load_data()

        confirm_button = r1col1.button("Confirm")
        if confirm_button:
            if st.session_state.cover_noise_image is not None:
                st.session_state.confirm_button = True
                with r1col2:
                    r2col2s = r1col2.columns([1, 3, 1])
                    watershed_img, hole_id, hole_areas, vertical_lengths = watershed_segmentation(
                        crop_and_resize(st.session_state.cover_noise_image),  # Import final img
                        crop_and_resize(st.session_state.recognition_image),  # Import ori-Cropped img
                        identification_ratio
                    )
                    r2col2s[1].image(watershed_img, caption='Watershed', use_column_width=True, channels="BGR",
                                     output_format="PNG")
                    # r2col2s[1].image("impage2.png", use_column_width=True, output_format="PNG")  # caption="Processed"
                    st.session_state.confirmed_image = watershed_img
                    st.session_state.hole_id = hole_id
                    st.session_state.hole_areas = hole_areas
                    st.session_state.vertical_lengths = vertical_lengths
                    st.write("You have confirmed the identification ratio: ", identification_ratio)

                if st.session_state.hole_id is not None and st.session_state.hole_areas is not None and st.session_state.vertical_lengths is not None:
                    st.session_state.df_csv = pd.DataFrame(
                        {'Slot ID': st.session_state.hole_id,
                         'Slot Size - Pixels': st.session_state.vertical_lengths,
                         'Slot Area - Pixels': st.session_state.hole_areas,
                         'Slot Size - mm [Manual]': [''] * len(st.session_state.hole_id),
                         'Slot Size - mm [Calculated]': [''] * len(st.session_state.hole_id),
                         'Slot Area - mm2 [Calculated]': [''] * len(st.session_state.hole_id)
                         }
                    )
        else:
            if st.session_state.confirmed_image is not None:
                with r1col2:
                    r2col2s = r1col2.columns([1, 3, 1])
                    r2col2s[1].image(st.session_state.confirmed_image, caption='Watershed', use_column_width=True, channels="BGR",
                                     output_format="PNG")
                    st.write("You have confirmed the identification ratio: ", identification_ratio)


    # Add a button to confirm the selected value
    # st.markdown('---')

    # if 'analyse_button' not in st.session_state:
    #     st.session_state.analyse_button = False
    if 'show_table' not in st.session_state:  # show table
        st.session_state.show_table = False
    if 'normalise_clicked' not in st.session_state:
        st.session_state.normalise_clicked = False
    if 'data_ratio' not in st.session_state:
        st.session_state.data_ratio = None
    if 'total_area' not in st.session_state:
        st.session_state.total_area = None
    if 'my_bar' not in st.session_state:
        st.session_state.my_bar = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # 创建一个固定的容器来放置进度条
    progress_container = st.empty()

    if st.session_state.confirmed_image is not None:
        if not st.session_state.analysis_complete:
            st.session_state.show_table = True

            # 使用固定容器来显示进度条
            with progress_container:
                progress_bar = st.progress(0, text="Analysing...")
                for percent_complete in range(1, 3):
                    time.sleep(1)
                    progress_bar.progress(percent_complete / 2, text="Analysing...")
                progress_bar.progress(100, text="Analysis Complete")

            st.session_state.analysis_complete = True
        else:
            # 如果分析已完成，只显示完成的进度条
            with progress_container:
                st.progress(100, text="Analysis Complete")

            st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                            ' color:Black; font-size: 15px; ">Grate slots size results</p></nobr>', unsafe_allow_html=True)

            # size_results_df = pd.read_csv('./Grate slots size results.csv', sep=',')

        if st.session_state.show_table:
            edited_df = st.data_editor(
                st.session_state.df_csv,
                key="data_editor",
                height=434,
                disabled=["Slot ID", "Slot Size - Pixels", "Slot Area - Pixels", "Slot Size - mm [Calculated]", "Slot Area - mm2 [Calculated]"],
                hide_index=True,
                num_rows="fixed",
            )
            is_equal_manual = edited_df["Slot Size - mm [Manual]"].equals(
                st.session_state.df_csv["Slot Size - mm [Manual]"])
            if not is_equal_manual:
                st.session_state.df_csv = edited_df  # update the variable
                st.experimental_rerun()

            areacol1, areacol2 = st.columns([1, 1], gap='small')

            with areacol1:
                if st.button("Normalise Measurements"):
                    st.session_state.normalise_clicked = True
                    # left_ratio, left_regularized = normalise_data_left(
                    #     st.session_state.df_csv["Left Port Size-mm(Measurements)"]
                    # )
                    st.session_state.df_csv, total_area, data_ratio = normalise_data_right(st.session_state.df_csv)

                    # st.session_state.left_ratio = left_ratio
                    st.session_state.data_ratio = data_ratio
                    st.session_state.total_area = total_area
                    is_equal_size = edited_df["Slot Size - mm [Calculated]"].equals(
                        st.session_state.df_csv["Slot Size - mm [Calculated]"])
                    is_equal_area = edited_df["Slot Area - mm2 [Calculated]"].equals(
                        st.session_state.df_csv["Slot Area - mm2 [Calculated]"])
                    if not is_equal_size or not is_equal_area:
                        # st.session_state.df_csv = edited_df  # update the variable
                        st.experimental_rerun()
                    # # 更新DataFrame
                    # st.session_state.df_csv['Left Port Size-mm(AI Tool)'] = left_regularized
                    # st.session_state.df_csv['Right Port Size-mm(AI Tool)'] = right_regularized
                    # 应用正则化
            if st.session_state.normalise_clicked:
                with areacol2:
                    areacol2.metric(label="Total Modified Open Area is [mm²]",
                                    value=round(st.session_state.total_area))
                st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                            ' color:Black; font-size: 15px; ">Please input grate/pebble service details:</p></nobr>',
                            unsafe_allow_html=True)
                r3col1, r3col2, r3col3 = st.columns(3)
                process_tons = r3col1.text_input('Total tons milled:', '0')
                process_days = r3col2.text_input('Total days since reline:', '0')
                r3col3.markdown('###')
                save_button = r3col3.button('Save into Database')
                if save_button:
                    if save_button:
                        try:
                            tons = float(process_tons)
                            days = float(process_days)

                            if tons > 0 and days > 0:
                                today = datetime.date.today().strftime("%d-%b-%Y")
                                df_csv_sheet2 = pd.DataFrame({
                                    'Export Date': [today],
                                    'Total Tons Milled': [tons],
                                    'Total Days Since Reline': [days],
                                    'Total Areas': [round(st.session_state.total_area)]
                                })
                                filename = f'{hole_layout}_{today}.xlsx'
                                # Assuming 'dataframe' and 'dataframe2' are already defined
                                with pd.ExcelWriter('./Data_Export/' + filename) as writer:
                                    st.session_state.df_csv.to_excel(writer, sheet_name='Grate slots size results', index=False)
                                    df_csv_sheet2.to_excel(writer, sheet_name='Service details', index=False)
                                st.success("Data exported to Excel successfully!")
                            else:
                                st.warning("Please enter non-zero values for tons and days.")
                        except ValueError:
                            st.error("Please enter valid numeric values for tons and days.")
                # process_tons = r3col3.text_input('Processed million tons:', '0')
                # process_tons = r3col3.number_input("Processed tons:", step=1, format="%.3f")

                # st.markdown('###')
            else:
                with areacol2:
                    areacol2.metric(label="Total AI calculated Open Area is [mm²]",
                                    value=round(st.session_state.df_csv["Slot Area - Pixels"].sum() * 3.92))

    hide_streamlit_style = """
                <style>
                tbody th {display:none}
                .blank{
                display: none;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def calculate_recycle_rate(pebble_number, total_tons):
    # Define the data for each pebble number
    data = {
        9: [(0, 358), (3, 923), (3.45, 999)],
        8: [(0, 340), (3, 876), (3.45, 947)],
        7: [(0, 321), (3, 828), (3.45, 896)],
        6: [(0, 303), (3, 780), (3.45, 844)],
        5: [(0, 284), (3, 733), (3.45, 793)],
        4: [(0, 266), (3, 685), (3.45, 741)],
        3: [(0, 247), (3, 638), (3.45, 690)],
        2: [(0, 229), (3, 590), (3.45, 638)],
    }

    # Extract the corresponding data for the given pebble number
    if pebble_number in data:
        tons, rates = zip(*data[pebble_number])

        # Fit the first linear function (from first point to second point)
        coef1 = np.polyfit(tons[:2], rates[:2], 1)

        # Fit the second linear function (from second point to third point)
        coef2 = np.polyfit(tons[1:], rates[1:], 1)

        # Calculate the predicted rate based on the total tons
        if total_tons <= tons[1]:
            predicted_rate = coef1[0] * total_tons + coef1[1]
        else:
            predicted_rate = coef2[0] * total_tons + coef2[1]

        return predicted_rate
    else:
        return None


def analysis_app():
    titleContainer = st.container()
    with titleContainer:
        titleColmns1, titleColmns2 = st.columns([3.2, 1.0], gap='small')
        with titleColmns1:
            st.title("Bradken - BHP ODO")
            st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                        ' color:Black; font-size: 24px; ">Svedala Discharge End Open Area AI </p></nobr>',
                        unsafe_allow_html=True)
        with titleColmns2:
            st.image('bradken.png')
            # st_lottie(lottie_coding, height=150, key="hello")

    st.markdown("-------------------")
    # Full Open Area Calculation
    st.markdown('<nobr><p style="text-align: left;font-family:sans serif;'
                ' color:Black; font-size: 15px; ">Total Open Area Analysis:</p></nobr>', unsafe_allow_html=True)
    # 获取.\\Data_Export目录下的所有Excel文件
    excel_files = [f for f in os.listdir('./Data_Export') if f.endswith('.xlsx')]

    # 提取所有不重复的日期
    dates = set()
    for file in excel_files:
        date_str = file.split('_')[-1].split('.')[0]
        try:
            date = datetime.datetime.strptime(date_str, '%d-%b-%Y')
            dates.add(date.strftime('%d-%b-%Y'))
        except ValueError:
            continue

    # 将日期排序
    sorted_dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y'), reverse=True)

    r4col1, r4col2 = st.columns([2.15, 1.0], gap='medium')
    # 添加日期选择框
    selected_date = r4col1.selectbox('Select Date:', sorted_dates)
    tons_data = []
    de_container = st.container()
    with de_container:
        r5col1, r5col2, r5col3 = st.columns(3)
        miss_data_type = []

        with r5col1:
            file_22mm = f'22mm Grate_{selected_date}.xlsx'
            file_22mm_path = os.path.join('./Data_Export', file_22mm)
            total_days1, total_area1, total_tons1 = get_total_days(file_22mm_path)
            if os.path.exists(file_22mm_path):
                # install_date1 = r5col1.metric('22mm Outer Grate Inspection:', total_days1)
                tons_data.append(total_tons1)
            else:
                # install_date1 = r5col1.metric('22mm Outer Grate Inspection:', 'No data',)
                miss_data_type.append('22mm Outer Grate')

            install_num1 = r5col1.text_input('22mm Outer Grate Number:', '0')

        with r5col2:
            file_65mm = f'65mm Pebble Grate_{selected_date}.xlsx'
            file_65mm_path = os.path.join('./Data_Export', file_65mm)
            total_days2, total_area2, total_tons2 = get_total_days(file_65mm_path)
            if os.path.exists(file_65mm_path):
                # install_date2 = r5col2.metric('65mm Pebble Grate Inspection:', total_days2)
                tons_data.append(total_tons2)
            else:
                # install_date2 = r5col2.metric('65mm Pebble Grate Inspection:', 'No data',)
                miss_data_type.append('65mm Pebble Grate')
            install_num2 = r5col2.text_input('65mm Pebble Grate Number:', '0')

        with r5col3:
            file_22mm_middle = f'22mm middle Grate_{selected_date}.xlsx'
            file_22mm_middle_path = os.path.join('./Data_Export', file_22mm_middle)
            total_days3, total_area3, total_tons3 = get_total_days(file_22mm_middle_path)
            if os.path.exists(file_22mm_middle_path):
                # install_date3 = r5col3.metric('22mm Middle Grate Inspection:', total_days3)
                tons_data.append(total_tons3)
            else:
                # install_date3 = r5col3.metric('22mm Middle Grate Inspection:', 'No data',)
                miss_data_type.append('22mm Middle Grate')
            install_num3 = r5col3.text_input('22mm Middle Grate Number:', '0')

    if len(tons_data) !=0:
        r4col2.metric('Total Tons Milled:', tons_data[0])

    st.markdown("###")
    OA_button = st.button("Calculate Total Open Area and Predict Recycle Rate")

    if OA_button:
        # st.write ="Analysing..."
        # progress_container = st.empty()
        my_bar = st.progress(0, text="Analysing...")
        for percent_complete in range(1, 3):
            time.sleep(1)
            my_bar.progress(percent_complete / 2, text="Calculating...")
        # progress_container.empty()

        #########################################################################
        r6col1, r6col2 = st.columns(2)
        if len(miss_data_type) == 0:
            try:
                outer_num = int(install_num1)
                pebble_num = int(install_num2)
                middle_num = int(install_num3)
                if outer_num > 0 and pebble_num > 0 and middle_num > 0:
                    if 1 < pebble_num <= 9:
                        r6col1.metric("Discharge End Total Open Area [m²]",
                                      round((total_area1*outer_num + total_area2*pebble_num + total_area3*middle_num)/1000000, 2))
                        if len(tons_data) != 0:
                            r6col2.metric("Predicted Recycle Rate [tph]", round(calculate_recycle_rate(pebble_num, max(tons_data)), 2))
                        else:
                            r6col2.warning(f"Missing 'Total Tons Milled' data in {sorted_dates}")
                    else:
                        st.warning("The Pebble Grate Number needs to be between 2 and 9")

                else:
                    st.warning("Please enter non-zero values for Grate/Pebble Number.")
            except ValueError:
                st.error("Please enter valid numeric values for Grate/Pebble Number.")
        else:
            formatted_str = ', '.join(miss_data_type)
            r6col2.error(f"Missing {formatted_str} data in {sorted_dates}")
        # plot_container = st.container()
        # with plot_container:
        #    r6col1, r6col2, r6col3 = st.columns(3)
        st.markdown("###")
        # OA_plot = HISTORICAL_DATA_PLOT(['15-03-2023', '12-04-2023', '15-05-2023'], [8.03, 8.78, 9.58])
        # st.plotly_chart(OA_plot, use_container_width=True)

    else:
        st.markdown('###')


if __name__ == "__main__":
    main()



