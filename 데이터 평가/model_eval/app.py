import streamlit as st
import os, json
import tempfile
import subprocess
import pandas as pd
from io import BytesIO
from functions.visualize import load_eval_results, get_mean_scores, plot_radar_chart_multi, plot_score_distribution, show_mean_score_table
from functions.feature_count import get_data_distribution
from functions.filtering import filter_jsonl_bytes_by_threshold

# =========================
# 캐시 폴더 관련 함수
# =========================
CACHE_DIR = "./cache"

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def save_eval_to_cache(filename: str, eval_bytes: bytes, mean_scores: dict, dist: dict, data_bytes: bytes):
    ensure_cache_dir()
    base = os.path.splitext(filename)[0]
    with open(os.path.join(CACHE_DIR, f"{base}_eval.jsonl"), "wb") as f:
        f.write(eval_bytes)
    with open(os.path.join(CACHE_DIR, f"{base}_mean.json"), "w", encoding="utf-8") as f:
        json.dump(mean_scores, f, ensure_ascii=False)
    with open(os.path.join(CACHE_DIR, f"{base}_dist.json"), "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False)
    with open(os.path.join(CACHE_DIR, f"{base}_data.jsonl"), "wb") as f:
        f.write(data_bytes)

def load_all_cached_files():
    ensure_cache_dir()
    cached = {}
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith("_eval.jsonl"):
            base = fname.replace("_eval.jsonl", "")
            try:
                with open(os.path.join(CACHE_DIR, f"{base}_eval.jsonl"), "rb") as f:
                    eval_bytes = f.read()
                with open(os.path.join(CACHE_DIR, f"{base}_mean.json"), "r", encoding="utf-8") as f:
                    mean_scores = json.load(f)
                with open(os.path.join(CACHE_DIR, f"{base}_dist.json"), "r", encoding="utf-8") as f:
                    dist = json.load(f)
                with open(os.path.join(CACHE_DIR, f"{base}_data.jsonl"), "rb") as f:
                    data_bytes = f.read()
                cached[f"{base}.jsonl"] = {
                    "data": data_bytes,
                    "eval": eval_bytes,
                    "mean_scores": mean_scores,
                    "dist": dist
                }
            except Exception as e:
                continue
    return cached

# Streamlit 업로드-캐시-세션 관리 
def remove_file_from_cache(filename):
    base = os.path.splitext(filename)[0]
    if filename in st.session_state["cached_files"]:
        del st.session_state["cached_files"][filename]
    for suffix in ["_eval.jsonl", "_mean.json", "_dist.json", "_data.jsonl"]:
        try:
            os.remove(os.path.join(CACHE_DIR, f"{base}{suffix}"))
        except Exception:
            pass

# =========================
# Streamlit 앱 시작
# =========================
st.title("말투변환 모델 성능 테스트")

# --- 캐시 폴더에서 기존 평가 결과 자동 로딩 ---
if "cached_files" not in st.session_state:
    st.session_state["cached_files"] = load_all_cached_files()

# 규칙 보기 toggle
if st.toggle("데이터 업로드 규칙 보기(업로드 중 금지)"):
    st.markdown("""
    ### 데이터 업로드 규칙

    #### 1. **파일 포맷**
    - **형식:** `.jsonl` (JSON Lines, 한 줄에 하나의 JSON 객체)
    - **인코딩:** `UTF-8`

    #### 2. **필수 필드 및 예시**
    | 필드명                | 설명                        | 예시 값                      |
    |-----------------------|-----------------------------|------------------------------|
    | `post_type`           | 동물 유형                   | `"cat"`, `"dog"`             |
    | `emotion`             | 감정                        | `"normal"`, `"happy"`, `"sad"`, `"angry"`, `"grumpy"`, `"curious"` |
    | `content`             | 입력 문장                   | `"오늘은 뭘 하고 놀까?"`      |
    | `transformed_content` | 변환(출력) 문장             | `"멍. 오늘 뭐하고 놀지? ..."`  |

    #### 3. **JSONL 예시**
    ```json
    {
    "post_type": "dog",
    "emotion": "normal",
    "content": "오늘은 뭘 하고 놀까?",
    "transformed_content": "멍. 오늘 뭐하고 놀지? 빨리 놀고 싶다멍. 🐾"
    }
    ```

    #### 4. **평가 대상 동물/감정**
    - **동물(post_type):**  
    - `cat` (고양이)  
    - `dog` (강아지)
    - **감정(emotion):**  
    - `normal` (일반)  
    - `happy` (기쁨)  
    - `sad` (슬픔)  
    - `angry` (분노)  
    - `grumpy` (까칠)  
    - `curious` (호기심)

    #### 5. **업로드 시 주의사항**
    - 각 줄마다 하나의 JSON 객체만 포함해야 합니다.
    - 모든 필드는 반드시 포함되어야 하며, 값이 없으면 빈 문자열로 입력하세요.
    - 파일 크기는 10MB 이하를 권장합니다.
    - 평가 점수(`kobert_score`, `type_score`, `quality_score`, `bleu_score`, `perplexity_score` 등)는 평가가 끝난 파일에만 포함됩니다.

    #### 6. **평가 기준 및 의미**
    | 점수명              | 의미                                                             | 특징     |
    |-------------------|-----------------------------------------------------------------|---------|
    | kobert_score      | KoBERT 기반 의미 유사도 (원문과 변환문장의 의미가 얼마나 유사한지 평가)        | 1에 가까울수록 의미가 유사함 (최대 1) |
    | type_score        | 말투/스타일 적합성 (동물/감정별 요구 조건에 맞는지 평가)                     | 1에 가까울수록 말투/스타일이 적합 (최대 1) |
    | quality_score     | 품질(의미 보존, 스타일 일치, 자연스러움, 형식 적합성 등 변환문장 품질 평가)      | 1에 가까울수록 품질이 우수 (최대 1) |
    | bleu_score        | BLEU 점수 (원문과 변환문장 간 n-gram 기반 유사도, 기계번역 품질 지표)        | `bleu_score / 0.1` 로 10배 값 사용(최대 1) |
    | perplexity_score  | 문장 자연스러움 (언어모델 기반 Perplexity, 낮을수록 자연스러운 문장)          | `0.1 / perplexity_score` 로 역수 및 1/10값 사용(최대 1) |

    - **모든 점수는 0~1로 변환되어 시각화 및 필터링에 사용됩니다.**

    #### 데이터셋 
    | 데이터셋 파일명                | 주요 특징 및 설명  | 데이터 개수 | 
    |-------------------------------|----------------------------------------|--------------|
    | test_made.jsonl                | 테스트용 Google AI Studio 합성 데이터.              | 100개 |
    | dataset_0515_made.jsonl        | 초기 유저 데이터.                              | 342개 |
    | dataset_0527_made.jsonl        | 유저 게시글 데이터 기반 감정별/동물별 데이터            | 818개 |
    | dataset_0530_made.jsonl        | 유저 게시글 데이터 기반 감정별/동물별 데이터 감정별 증폭   | 2,986개 |
    | dataset_0613_made.jsonl        | 유저 댓글 입력에 대한 규칙기반 변환(cat) 데이터         | 681개 | 
    | dataset_0620_made.jsonl        | 유저 댓글 입력에 대한 규칙기반 변환(dog) 데이터          | 681개 | 
    | dataset_0622_made.jsonl        | 합성 인풋에 대한 Gemini 말투 변환 최종본            | 17,596개 | 
    | dataset_0629_all.jsonl        | 이전까지의 모든 데이터를 합친 통합 데이터.             | 21,104개 | 
    | dataset_0709_made.jsonl        | 많이 사용되는 비문 데이터 및 safty 데이터 추가 | - 개 | 
    | dataset_0710_made.jsonl        | normal 데이터에 대해 댓글 데이터로만 사용     | - 개 | 
    | dataset_0710_all.jsonl        | 0709 + 0710 데이터셋                     | - 개 | 
    ---
    **예시 파일을 참고하여 동일한 구조로 데이터를 준비해 주세요.**
    문제가 있으면 담당자에게 문의 바랍니다.
        """)

metric_labels = {
    "kobertscore_f1": "KoBERT",
    "type_score": "Type",
    "quality_score": "Quality",
    "bleu_score": "BLEU",
    "perplexity_score": "Perplexity"
}

all_metrics = list(metric_labels.keys())

# --- 업로드 파일 및 평가 결과 캐싱 ---
uploaded_files = st.file_uploader(
    "여러 개의 JSONL 데이터 파일을 업로드하세요. (분포 통계 및 모델 평가 자동 진행)",
    type=["jsonl"],
    accept_multiple_files=True,
    key="data_and_eval"
)

# 새로 업로드된 파일을 캐시에 저장 및 평가/통계 수행
if uploaded_files:
    for f in uploaded_files:
        fname = f.name
        if not fname or not isinstance(fname, str):
            st.error("잘못된 파일명입니다. 파일을 다시 업로드 해주세요.")
            continue
        if fname not in st.session_state["cached_files"]:
            remove_file_from_cache(fname)
            st.session_state["cached_files"][fname] = {"data": f.getvalue(), "eval": None, "mean_scores": None, "dist": None}
            dist = get_data_distribution(f.getvalue())
            st.session_state["cached_files"][fname]["dist"] = dist
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_file:
                tmp_file.write(f.getvalue())
                tmp_path = tmp_file.name
            eval_path = tmp_path.replace(".jsonl", "_eval.jsonl")
            main_eval_path = os.path.join(os.path.dirname(__file__), "functions", "main_eval.py")
            with st.spinner(f"모델 평가 수행 중: {fname}"):
                result = subprocess.run(
                    [
                        "python", main_eval_path,
                        "--input_path", tmp_path,
                        "--output_path", eval_path,
                        "--use_kobert", "--use_type", "--use_quality", "--use_bleu", "--use_perplexity"
                    ],
                    capture_output=True, text=True
                )
            if result.returncode != 0:
                st.error(f"평가 실패: {fname}\n{result.stderr}")
                continue
            with open(eval_path, "rb") as f_eval:
                eval_jsonl_bytes = f_eval.read()
                st.session_state["cached_files"][fname]["eval"] = eval_jsonl_bytes
            results = load_eval_results(eval_path)
            mean_scores = get_mean_scores(results, all_metrics)
            st.session_state["cached_files"][fname]["mean_scores"] = mean_scores
            st.success(f"✅ {fname} 평가 및 통계 완료!")
            save_eval_to_cache(
                fname,
                eval_jsonl_bytes,
                mean_scores,
                dist,
                f.getvalue()
            )

# 캐시 파일명 리스트 생성 시 문자열만 포함
cached_file_names = [k for k in st.session_state["cached_files"].keys() if isinstance(k, str) and k]

selected_cached_files = st.multiselect(
    "캐시에 저장된 파일을 선택해서 불러올 수 있습니다.",
    cached_file_names,
    default=[],
)

st.markdown("#### 평가 결과 파일(jsonl) 다운로드")
download_file = st.selectbox(
    "다운로드할 평가 결과 파일을 선택하세요.",
    cached_file_names
)
if download_file:
    eval_bytes = st.session_state["cached_files"][download_file].get("eval")
    if eval_bytes is not None:
        st.download_button(
            label=f"다운로드",
            data=eval_bytes,
            file_name=f"{os.path.splitext(download_file)[0]}_eval.jsonl",
            mime="application/json"
        )
    else:
        st.warning("다운로드할 데이터가 없습니다.")

file_objs = []
for fname in selected_cached_files:
    if not isinstance(fname, str):
        st.error(f"잘못된 fname: {fname} (type: {type(fname)})")
        continue
    file_obj = BytesIO(st.session_state["cached_files"][fname]["data"])
    file_obj.name = fname
    file_objs.append(file_obj)

scores_list = []
model_names = []
table_rows = []

if file_objs:
    st.markdown("-----------------------")
    for uploaded_file in file_objs:
        fname = uploaded_file.name
        dist = st.session_state["cached_files"][fname]["dist"]
        st.markdown(f"#### ⬇️ {fname} 데이터 분포 (동물별/감정별)")
        pivot_data = []
        for post_type, emotion_counter in dist["type_emotion_counter"].items():
            row = {"post_type": post_type}
            for emotion in dist["emotion_order"]:
                row[emotion] = emotion_counter.get(emotion, 0)
            for emotion, count in emotion_counter.items():
                if emotion not in dist["emotion_order"]:
                    row[emotion] = count
            pivot_data.append(row)
        pivot_df = pd.DataFrame(pivot_data).set_index("post_type")
        st.dataframe(pivot_df, use_container_width=True)
        st.success(f"✅ 총 데이터 개수: {dist['total_count']} / 중복 없는 원문 개수: {dist['unique_content_count']}")

        model_name = os.path.splitext(fname)[0]
        model_names.append(model_name)
        mean_scores = st.session_state["cached_files"][fname]["mean_scores"]
        if mean_scores is None:
            mean_scores = {}
        table_row = {"모델명": model_name}
        table_row.update({metric_labels[metric]: mean_scores.get(metric, None) for metric in all_metrics})
        table_rows.append(table_row)
        scores_list.append(mean_scores)

    if table_rows:
        st.markdown("-----------------------")
        st.markdown("#### 시각화할 평가 지표 및 threshold 값을 선택하세요.")

        selected_metrics = []
        thresholds = {}
        cols = st.columns(len(all_metrics))

        default_thresholds = {
            "kobertscore_f1": 0.6,
            "type_score": 0.8,
            "quality_score": 0.8,
            "bleu_score": 0.2,
            "perplexity_score": 0.5,
        }
        for i, metric in enumerate(all_metrics):
            label = metric_labels[metric]
            with cols[i]:
                checked = st.session_state.get(f"metric_{metric}", True)
                if st.checkbox(label, value=checked, key=f"metric_{metric}"):
                    selected_metrics.append(metric)
                    thresholds[metric] = st.number_input(
                        f"{label} threshold", min_value=0.0, max_value=1.0,
                        value=default_thresholds.get(metric, 0.5), step=0.01, key=f"thres_{metric}"
                    )

        if scores_list and selected_metrics:
            st.markdown("#### 선택한 지표로 레이더 차트 시각화")
            plot_radar_chart_multi(
                scores_list, model_names, selected_metrics,
                title="Evaluation Score", metric_labels=metric_labels, thresholds=thresholds
            )
            show_mean_score_table(scores_list, model_names, selected_metrics, metric_labels)


            st.markdown("#### 선택한 지표별 점수 분포 (히스토그램 & 박스플롯)")
            eval_jsonl_bytes_list = [
                st.session_state["cached_files"][fname]["eval"] for fname in selected_cached_files
            ]
            plot_score_distribution(
                eval_jsonl_bytes_list,
                [os.path.splitext(fname)[0] for fname in selected_cached_files],
                selected_metrics,
                metric_labels=metric_labels,
                thresholds=thresholds
            )

            st.markdown("-----------------------")
            st.markdown("#### 선택된 데이터 전체를 합쳐서 필터링 및 다운로드")
            if selected_cached_files:
                st.markdown("**선택된 데이터셋:** " + ", ".join(selected_cached_files))
            else:
                st.markdown("**선택된 데이터셋이 없습니다.**")

            thres_str = ", ".join([f"{metric_labels[m]}: {thresholds[m]}" for m in thresholds])
            st.success(f"✅ 적용된 threshold 값: {thres_str}")

            eval_jsonl_bytes_list = [
                st.session_state["cached_files"][fname]["eval"] for fname in selected_cached_files
            ]

            if st.button("필터링"):
                total_before = sum(
                    len(st.session_state["cached_files"][fname]["eval"].decode("utf-8").splitlines())
                    for fname in selected_cached_files
                )
                # threshold 기준 필터링
                filtered_data = filter_jsonl_bytes_by_threshold(eval_jsonl_bytes_list, thresholds)

                total_after = len(filtered_data)

                score_keys = set(metric_labels.keys())
                filtered_data_no_scores = []
                for d in filtered_data:
                    filtered = {k: v for k, v in d.items() if k not in score_keys}
                    filtered_data_no_scores.append(filtered)

                filtered_jsonl = "\n".join([json.dumps(d, ensure_ascii=False) for d in filtered_data_no_scores])

                st.write(f"필터링 전 데이터 개수: {total_before} → 필터링 후 데이터 개수: {total_after} (감소: {total_before - total_after})")

                st.markdown("#### 필터링된 데이터셋 분포 통계 (테이블)")
                from functions.feature_count import get_data_distribution
                filtered_jsonl_bytes = filtered_jsonl.encode("utf-8")
                dist = get_data_distribution(filtered_jsonl_bytes)
                st.success(f"✅ 총 데이터 개수: {dist['total_count']} / 중복 없는 원문 개수: {dist['unique_content_count']}")

                pivot_data = []
                for post_type, emotion_counter in dist["type_emotion_counter"].items():
                    row = {"post_type": post_type}
                    for emotion in dist["emotion_order"]:
                        row[emotion] = emotion_counter.get(emotion, 0)
                    for emotion, count in emotion_counter.items():
                        if emotion not in dist["emotion_order"]:
                            row[emotion] = count
                    pivot_data.append(row)
                pivot_df = pd.DataFrame(pivot_data).set_index("post_type")
                st.dataframe(pivot_df, use_container_width=True)

                st.download_button(
                    label="다운로드",
                    data=filtered_jsonl.encode("utf-8"),
                    file_name="filtered_all.jsonl",
                    mime="application/json"
                )
else:
    st.info("JSONL 파일을 업로드하거나, 캐시에서 파일을 선택하면 모델 평가와 데이터 분포를 확인할 수 있습니다.")