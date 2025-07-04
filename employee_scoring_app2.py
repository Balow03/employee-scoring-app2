import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import matplotlib as mpl
import matplotlib.font_manager as fm  # Import font_manager
from datetime import date

# --- Matplotlib 中文字体设置 (增强版) ---
# 尝试设置中文字体，优先使用用户系统可能存在的字体
# 增加了更多在Linux环境下常见的开源中文字体，如文泉驿系列
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'FangSong',  # Windows/常见字体
    'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC',  # Linux/通用中文字体
    'Arial Unicode MS'  # 某些系统可能有的通用Unicode字体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 强制Matplotlib刷新字体缓存，确保它能识别新设置的字体
# 这在部署环境中尤其重要，因为字体可能刚被安装或识别
fm.fontManager.rebuild()
# --- 字体设置结束 ---


# Define error types and their corresponding penalty points based on severity
ERROR_PENALTIES = {
    "安全隐患": 30,  # Extremely severe
    "未按规定处理废弃物": 15,  # High severity (environmental, safety)
    "未做好交接/记录": 12,  # High severity (accountability, traceability)
    "未挂标识牌": 10,  # Medium-High severity (safety, process control)
    "液封不合格": 10,  # Medium-High severity (safety, hygiene)
    "参数设置错误": 10,  # Medium-High severity (product quality, equipment damage)
    "操作顺序错误": 8,  # Medium severity (efficiency, quality)
    "遗漏清点": 8,  # Medium severity (inventory, quality)
    "放置不当": 7,  # Medium severity (organization, potential damage)
    "未及时完成": 7,  # Medium severity (efficiency)
    "清洁不彻底": 6,  # Low-Medium severity (hygiene, quality)
    "操作不规范": 5,  # Low severity (general deviation)
    "工具/器具未归位": 4,  # Low severity (organization, efficiency)
    "文件记录不规范": 3,  # Low severity (compliance, traceability)
    "其他错误": 5,  # Default penalty for unspecified errors
}

# List of predefined error options for the multiselect box (excluding "安全隐患" which is a checkbox)
PREDEFINED_ERROR_OPTIONS = [
    "遗漏清点", "放置不当", "未做好交接/记录", "未挂标识牌", "清洁不彻底",
    "未按规定处理废弃物", "工具/器具未归位", "液封不合格", "文件记录不规范",
    "操作顺序错误", "参数设置错误", "操作不规范", "未及时完成", "其他错误"
]

# Define predefined operation descriptions based on the provided document
PREDEFINED_OPERATION_DESCRIPTIONS = [
    {"item": "中间产品", "description": "清点、送规定地点放置，挂状态标识牌，与车间物料员做好交接并在记录上签字"},
    {"item": "样品瓶", "description": "清点数量后交车间物料员退回中转站保存"},
    {"item": "废弃物", "description": "清离现场、置垃圾暂存处销毁"},
    {"item": "文件记录", "description": "与后续产品无关的清离现场"},
    {"item": "工具器具", "description": "灭菌柜专用小车冲洗、湿抹、消毒或,清扫干净，置规定地点"},
    {"item": "废物贮器", "description": "冲洗、清扫干净，无积液，,并置规定地点，挂状态标识牌"},
    {"item": "生产设备", "description": "水浴式灭菌柜清洗消毒，无可见药物残留，,无油污，设备见本色，悬挂状态标识牌"},
    {"item": "工作场地", "description": "生产场所清扫、湿抹或湿拖干净，,悬挂状态标识符合状态要求"},
    {"item": "地漏", "description": "清洁、液封"},
    {"item": "清洁工具", "description": "清洗干净，置规定处存放，挂状态标识牌"},
]

# Format options for the selectbox: "清场项目: 操作要点"
FORMATTED_OPERATION_OPTIONS = [
    f"{op['item']}: {op['description']}" for op in PREDEFINED_OPERATION_DESCRIPTIONS
]


# Simulated Decision Tree/Random Forest logic for scoring
def calculate_score(completion_degree, error_types_list):
    """
    Calculates a single operation's score based on its completion degree and specific error types.

    Args:
        completion_degree (float): Completion degree for this specific operation (0-100).
        error_types_list (list): A list of error types encountered for this specific operation.

    Returns:
        int: The calculated score for this operation (0-100).
    """
    score = completion_degree

    total_error_penalty = 0
    for error_type in error_types_list:
        penalty = ERROR_PENALTIES.get(error_type, ERROR_PENALTIES.get("其他错误", 5))
        total_error_penalty += penalty

    score -= total_error_penalty

    # Adjust score based on completion degree thresholds and presence of severe errors
    has_safety_hazard_in_errors = "安全隐患" in error_types_list

    if completion_degree >= 95 and total_error_penalty == 0:
        score = 100
    elif completion_degree >= 90 and not has_safety_hazard_in_errors:
        score = max(85, score)
    elif completion_degree >= 80 and not has_safety_hazard_in_errors:
        score = max(75, score)
    elif completion_degree < 70:
        score = min(score, 60)

    return max(0, min(100, score))


def evaluate_all_operations(operations_data):
    """
    Calculates scores for each individual operation record.

    Args:
        operations_data (list of dict): Raw list of operation records.

    Returns:
        pandas.DataFrame: DataFrame with individual operation scores, including employee_id and date.
    """
    if not operations_data:
        return pd.DataFrame(columns=['employee_id', 'date', 'score'])

    scored_ops = []
    for op in operations_data:
        score = calculate_score(op['completion_degree'], op['error_types'])
        scored_ops.append({
            'employee_id': op['employee_id'],
            'date': op['date'],
            'score': score,
            'operation_description': op['operation_description'],  # Keep for context
            'error_types': op['error_types']  # Keep for context
        })
    return pd.DataFrame(scored_ops)


def generate_daily_performance_chart(daily_scores_df, employee_id):
    """
    Generates a line chart for a specific employee's daily performance.

    Args:
        daily_scores_df (pandas.DataFrame): DataFrame with daily average scores.
        employee_id (str): The ID of the employee to chart.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """
    employee_data = daily_scores_df[daily_scores_df['employee_id'] == employee_id].sort_values('date')

    if employee_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(employee_data['date'], employee_data['daily_avg_score'], marker='o', linestyle='-', color='blue')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('每日平均评分 (0-100)', fontsize=12)
    ax.set_title(f'{employee_id} 每日操作表现折线图', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def generate_overall_average_chart(overall_scores_df):
    """
    Generates a bar chart comparing overall average scores of all employees.

    Args:
        overall_scores_df (pandas.DataFrame): DataFrame with overall average scores per employee.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """
    if overall_scores_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(overall_scores_df['employee_id'], overall_scores_df['overall_avg_score'], color='lightgreen')
    ax.set_xlabel('员工ID', fontsize=12)
    ax.set_ylabel('总平均评分 (0-100)', fontsize=12)
    ax.set_title('所有员工总平均评分对比', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="车间人员操作评分系统")

st.title('🏭 车间人员操作评分系统')
st.markdown("""
    本系统旨在帮助管理者对员工的操作进行监督管理，提高生产效率和产品质量保障。
    请在下方输入员工的各项操作数据，系统将自动进行评分并生成图表。
""")

# Initialize session state for storing operations if not already present
if 'operations' not in st.session_state:
    st.session_state.operations = []  # Stores raw input data for each operation
if 'individual_op_scores_df' not in st.session_state:
    st.session_state.individual_op_scores_df = pd.DataFrame(
        columns=['employee_id', 'date', 'score'])  # Stores score for each individual operation
if 'daily_employee_scores_df' not in st.session_state:
    st.session_state.daily_employee_scores_df = pd.DataFrame(
        columns=['employee_id', 'date', 'daily_avg_score'])  # Stores daily average scores per employee
if 'overall_employee_scores_df' not in st.session_state:
    st.session_state.overall_employee_scores_df = pd.DataFrame(
        columns=['employee_id', 'overall_avg_score'])  # Stores overall average scores per employee

st.header('1. 输入操作数据')

with st.form("operation_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        employee_id = st.text_input("员工ID", key="employee_id_input")
        operation_date = st.date_input("操作日期", value=date.today(), key="op_date_input")
        selected_operation_description = st.selectbox(
            "选择操作要点描述",
            options=FORMATTED_OPERATION_OPTIONS,
            key="op_desc_selectbox"
        )
        operation_remark = st.text_input("操作要点备注 (可选)", key="op_remark_input")
    with col2:
        completion_degree = st.slider("完成度 (%)", 0, 100, 80, key="comp_deg_input")

        selected_error_types = st.multiselect(
            "选择错误类型 (可多选)",
            options=PREDEFINED_ERROR_OPTIONS,
            default=[],
            key="error_types_multiselect"
        )
        has_safety_hazard = st.checkbox("是否存在安全隐患", key="safety_hazard_input")

        other_error_remark = ""
        if "其他错误" in selected_error_types:
            other_error_remark = st.text_input("其他错误具体描述", key="other_error_remark_input")

    add_operation_button = st.form_submit_button("添加操作记录")

    if add_operation_button:
        if employee_id and selected_operation_description:
            final_error_types = list(selected_error_types)
            if has_safety_hazard:
                final_error_types.append("安全隐患")

            st.session_state.operations.append({
                'employee_id': employee_id,
                'date': operation_date,
                'operation_description': selected_operation_description,
                'operation_remark': operation_remark,
                'completion_degree': completion_degree,
                'error_types': final_error_types,
                'has_safety_hazard_display': '是' if has_safety_hazard else '否'
            })
            st.success(f"已添加 '{employee_id}' 在 {operation_date} 的操作记录。")
        else:
            st.warning("员工ID和操作要点描述不能为空。")

st.subheader('当前已添加的操作记录:')
if st.session_state.operations:
    display_df = pd.DataFrame(st.session_state.operations)
    display_df['错误类型'] = display_df['error_types'].apply(lambda x: ", ".join(x) if x else "无")
    display_df.rename(columns={
        'employee_id': '员工ID',
        'date': '日期',
        'operation_description': '操作要点描述',
        'operation_remark': '操作要点备注',
        'completion_degree': '完成度(%)',
        'has_safety_hazard_display': '是否存在安全隐患'
    }, inplace=True)
    st.dataframe(display_df[[
        '员工ID', '日期', '操作要点描述', '操作要点备注', '完成度(%)', '错误类型', '是否存在安全隐患'
    ]])
else:
    st.info("暂无操作记录。请添加记录。")

col_buttons = st.columns(2)
with col_buttons[0]:
    if st.button("清空所有记录"):
        st.session_state.operations = []
        st.session_state.individual_op_scores_df = pd.DataFrame(columns=['employee_id', 'date', 'score'])
        st.session_state.daily_employee_scores_df = pd.DataFrame(columns=['employee_id', 'date', 'daily_avg_score'])
        st.session_state.overall_employee_scores_df = pd.DataFrame(columns=['employee_id', 'overall_avg_score'])
        st.rerun()
with col_buttons[1]:
    if st.button("开始评分并生成图表"):
        if not st.session_state.operations:
            st.warning("没有可用于评分的数据。请先添加操作记录。")
        else:
            st.session_state.individual_op_scores_df = evaluate_all_operations(st.session_state.operations)

            if not st.session_state.individual_op_scores_df.empty:
                st.session_state.daily_employee_scores_df = \
                st.session_state.individual_op_scores_df.groupby(['employee_id', 'date'])['score'].mean().reset_index()
                st.session_state.daily_employee_scores_df.rename(columns={'score': 'daily_avg_score'}, inplace=True)
            else:
                st.session_state.daily_employee_scores_df = pd.DataFrame(
                    columns=['employee_id', 'date', 'daily_avg_score'])

            if not st.session_state.individual_op_scores_df.empty:
                st.session_state.overall_employee_scores_df = \
                st.session_state.individual_op_scores_df.groupby('employee_id')['score'].mean().reset_index()
                st.session_state.overall_employee_scores_df.rename(columns={'score': 'overall_avg_score'}, inplace=True)
            else:
                st.session_state.overall_employee_scores_df = pd.DataFrame(columns=['employee_id', 'overall_avg_score'])

            st.success("评分和图表数据已生成！")

st.header('2. 评分结果与图表分析')

if not st.session_state.individual_op_scores_df.empty:
    st.subheader('单次操作评分明细')
    st.dataframe(st.session_state.individual_op_scores_df[[
        'employee_id', 'date', 'score', 'operation_description', 'error_types'
    ]].rename(columns={
        'employee_id': '员工ID', 'date': '日期', 'score': '评分',
        'operation_description': '操作要点', 'error_types': '错误类型'
    }))

    st.subheader('员工每日平均表现折线图')
    all_employees = st.session_state.daily_employee_scores_df['employee_id'].unique()
    if len(all_employees) > 0:
        selected_employee_for_line_chart = st.selectbox(
            "选择要查看每日表现的员工",
            options=all_employees,
            key="select_employee_line_chart"
        )
        line_chart_fig = generate_daily_performance_chart(st.session_state.daily_employee_scores_df,
                                                          selected_employee_for_line_chart)
        if line_chart_fig:
            st.pyplot(line_chart_fig)
        else:
            st.info(f"员工 {selected_employee_for_line_chart} 暂无每日评分数据。")
    else:
        st.info("暂无员工每日评分数据。")

    st.subheader('所有员工总平均评分柱状图')
    bar_chart_fig = generate_overall_average_chart(st.session_state.overall_employee_scores_df)
    if bar_chart_fig:
        st.pyplot(bar_chart_fig)
    else:
        st.info("暂无员工总平均评分数据。")
else:
    st.info("点击 '开始评分并生成图表' 按钮以查看结果。")

st.markdown("---")
st.markdown("© 2025 车间人员操作评分系统")

