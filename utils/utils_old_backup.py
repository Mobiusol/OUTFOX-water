"""
重构后的工具模块 - 保持向后兼容
本文件现在主要作为重构后模块的统一入口点
保持 make_prompt_for_detection 和 make_prompt_for_watermark_generation 不变
"""

# 重要函数保持不变，确保向后兼容
from .base_utils import (
    load_pkl, save_pkl, json2dict, truncate_text, 
    string2token_nums, make_mixed_data
)

from .api_service import (
    completions_with_backoff, generation_by_qwen, 
    call_model_api, process_reply_from_qwen, remove_common_prefixes
)

from .embedding_service import (
    get_cached_embedding, get_comprehensive_similarity
)

from .metrics import (
    compute_three_recalls, compute_metrics
)

from .prompt_generator import (
    make_prompt_for_detection, make_prompt_for_watermark_generation,
    convert_to_detector_examples
)

from .confidence_calculators import (
    get_confidence_voting
)

from .watermark_validator import (
    identify_watermark_effectiveness
)

from .similarity_service import (
    find_similar_cases
)

from .config import COMMON_PREFIXES

# 向后兼容的全局变量
embedding_cache = {}

# 重新导出所有重要函数，确保向后兼容
__all__ = [
    'load_pkl', 'save_pkl', 'json2dict', 'truncate_text', 
    'string2token_nums', 'make_mixed_data',
    'completions_with_backoff', 'generation_by_qwen', 
    'call_model_api', 'process_reply_from_qwen', 'remove_common_prefixes',
    'get_cached_embedding', 'get_comprehensive_similarity',
    'compute_three_recalls', 'compute_metrics',
    'make_prompt_for_detection', 'make_prompt_for_watermark_generation',
    'convert_to_detector_examples',
    'get_confidence_voting',
    'identify_watermark_effectiveness',
    'find_similar_cases',
    'COMMON_PREFIXES'
]

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def json2dict(path):
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_model_and_tokenizer(name):
    """加载模型和tokenizer，如果T5不可用则返回None"""
    if not T5_AVAILABLE:
        print(f"Warning: T5 not available, cannot load {name}")
        return None, None

    try:
        model = T5ForConditionalGeneration.from_pretrained(name)
        tokenizer = T5Tokenizer.from_pretrained(name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        return None, None

def truncate_text(text, truncated_length, truncate_type):
    if truncate_type == 'token':
        truncated_encodes = qwen_encoding.encode(text)[:truncated_length]
        return qwen_encoding.decode(truncated_encodes)
    elif truncate_type == 'word':
        return ' '.join(text.split(' ')[:truncated_length])

def string2token_nums(string):
    num_tokens = 0
    num_tokens = len(qwen_encoding.encode(string))
    return num_tokens

def make_mixed_data(human_path, lm_path, ps_path):
    random.seed(42)
    humans, lms, pss = load_pkl(human_path), load_pkl(lm_path), load_pkl(ps_path)
    humans_with_label_ps, lms_with_label_ps = [(human, '0', ps) for human, ps in zip(humans, pss)], [(lm, '1', ps) for lm, ps in zip(lms, pss)]
    all_with_label_ps = humans_with_label_ps + lms_with_label_ps
    random.shuffle(all_with_label_ps)
    data = [t[0] for t in all_with_label_ps]
    labels = [t[1] for t in all_with_label_ps]
    pss = [t[2] for t in all_with_label_ps]
    return (data, labels, pss)

@backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=5)
def completions_with_backoff(**kwargs):
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",  # 固定模型名称
        "input": {
            "prompt": kwargs.get("prompt"),
            "messages": kwargs.get("messages")
        },
        "parameters": {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
    }
    response = requests.post(DASHSCOPE_ENDPOINT, headers=headers, json=payload)
    result = response.json()
    return result["output"]["text"]

def remove_common_prefixes(text):
    """移除常见前缀的统一函数"""
    processed_text = text.strip()
    for prefix in COMMON_PREFIXES:
        if processed_text.startswith(prefix):
            processed_text = processed_text.replace(prefix, "", 1).strip()
            break
    return processed_text

COMMON_PREFIXES = [
    "Here's the watermarked text embedding detectable semantic constructs:",
    "Here is the watermarked text:",
    "Watermarked:",
    "Certainly! Here's the watermarked text:",
    "---",
    "Here is the watermarked text embedding with detectable semantic constructs:",
    "Certainly, here’s the watermarked version of the provided text:",
    "Certainly! Here’s the watermarked version of the text:",
    "Of course. Below is the watermarked version:",
    "Certainly! Here's a watermarked version of your text:",
]

def generation_by_qwen(prompt, original_text_tokens):
    lm_essay = ''
    while True:
        try:
            # 如果 prompt 过长，可以考虑 truncate 或分段处理
            lm_essay = completions_with_backoff(
                prompt=prompt,
                temperature=0.8,
                top_p=1,
                max_tokens=original_text_tokens + 50
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            time.sleep(3)

    # 使用统一的前缀移除逻辑
    processed_text = remove_common_prefixes(lm_essay)
    return processed_text

def call_model_api(prompt):
    res = completions_with_backoff(
        prompt=prompt,
        temperature=0.1,
        top_p=0.9,
        max_tokens=10
    )
    return res.strip().lower()

def make_prompt_for_detection(target_text, examples):
    instruction = """Classify whether the text is Original or contains an implicit Watermark.
    An implicit watermark is NOT visible to humans but can be detected through patterns in word choice, syntactic structures, or semantic patterns.
    IMPORTANT: You must answer with ONLY the single word 'Original' OR 'Watermarked'. Any other format will be considered incorrect.\n"""

    prompt = instruction

    # Calculate token limits to stay within API constraints
    example_overhead = string2token_nums("Text: \nAnswer: Original\n")
    target_overhead = string2token_nums(f"Text: {target_text}\nAnswer: ")
    instruction_tokens = string2token_nums(instruction)

    # Calculate available tokens for examples
    available_tokens = 10000 - (instruction_tokens + target_overhead)
    tokens_per_example = available_tokens // len(examples) if examples else 0

    # Shuffle examples for better prompt diversity
    random_examples = examples.copy()
    random.shuffle(random_examples)

    # Add examples with truncation if needed
    for ex in random_examples:
        text = truncate_text(ex['text'], tokens_per_example - example_overhead, 'token') if tokens_per_example > example_overhead else ex['text']
        prompt += f"Text: {text}\nAnswer: {ex['label']}\n"

    # Add target text
    prompt += f"Text: {target_text}\nAnswer: "
    return prompt

# 将生成器的输出转换为检测器所需的格式
def convert_to_detector_examples(gen_examples):
    detector_examples = []
    for ex in gen_examples["success"]:
        detector_examples.append({"text": ex["original"], "label": "Original"})
        detector_examples.append({"text": ex["watermarked"], "label": "Watermarked"})
    for ex in gen_examples["failure"]:
        detector_examples.append({"text": ex["watermarked"], "label": "Watermarked"})  # 失败案例仍标记为Watermarked
    random.shuffle(detector_examples)
    return detector_examples

def make_prompt_for_watermark_generation(original_text, examples, max_examples=10):
    instruction_1 = """Your response MUST start directly with the watermarked text.
    DO NOT include any introduction, explanation, or prefix like "Here's", "Certainly", etc.
    ANY prefix will be considered a critical error.

    Generate a text similar to the Original Text below, but embed COMPLETELY IMPLICIT watermark patterns.

    IMPORTANT WATERMARK RULES:
    1. MUST make noticeable changes - do not copy the original text
    2. The watermark must be purely implicit and undetectable to humans
    3. NEVER add ANY explicit markers, tags, symbols, formatting or identifiable text patterns
    4. The watermarked text must appear completely natural to human readers
    5. The watermarked text must maintain VERY HIGH semantic similarity to the original
    6. Preserve the original meaning, tone, and main ideas exactly

    IMPORTANT: Start your response directly with the watermarked text without any introduction or prefix.\n"""

    instruction_2 = f"\nNow generate a watermarked text that is highly similar to the original:\nOriginal: {original_text}\nWatermarked: "
    original_text_tokens = string2token_nums(original_text)

    # 如果没有成功案例，返回基本提示
    if len(examples["success"]) == 0:
        return instruction_1 + instruction_2

    # 计算token限制
    instruction_tokens = string2token_nums(instruction_1 + instruction_2)
    example_format_tokens = string2token_nums("Original: \nWatermarked: \n\n") * min(len(examples["success"]), max_examples)

    # 计算可用于案例的token数
    available_tokens = 10000 - (instruction_tokens + example_format_tokens + original_text_tokens + 50)
    truncated_length = available_tokens // (2 * min(len(examples["success"]), max_examples))

    # 基于相似度选择最相关的案例，并优先选择高相似度案例
    similar_cases = find_similar_cases(original_text, examples["success"], top_k=max_examples)

    # 构建提示
    prompt = instruction_1
    prompt += "Examples of successful watermarked texts with high similarity (detected):\n"

    # 添加相似案例，优先显示高相似度的案例
    for case in similar_cases:
        try:
            # 兼容不同的数据结构：EnhancedCase对象或字典
            if hasattr(case, 'original'):
                # 如果是EnhancedCase对象
                case_original = case.original
                case_watermarked = case.watermarked
                case_confidence = getattr(case, 'confidence', 0.5)
            elif isinstance(case, dict):
                # 如果是字典
                case_original = case.get('original', '')
                case_watermarked = case.get('watermarked', '')
                case_confidence = case.get('confidence', 0.5)
            else:
                print(f"跳过未知格式的案例: {type(case)}")
                continue

            if not case_original or not case_watermarked:
                continue

            # 计算相似度并在示例中显示
            similarity = get_comprehensive_similarity(case_original, case_watermarked)
            original_truncated = truncate_text(case_original, truncated_length, 'token')
            watermarked_truncated = truncate_text(case_watermarked, truncated_length, 'token')

            prompt += f"Original: {original_truncated}\n"
            prompt += f"Watermarked: {watermarked_truncated}\n"
            prompt += f"(Similarity: {similarity:.2f}, Confidence: {case_confidence:.2f})\n\n"
        except Exception as e:
            # 如果出错，使用简单方法
            print(f"处理案例时出错: {e}")
            if hasattr(case, 'original'):
                original_truncated = truncate_text(case.original, truncated_length, 'token')
                watermarked_truncated = truncate_text(case.watermarked, truncated_length, 'token')
                case_confidence = getattr(case, 'confidence', 0.5)
            elif isinstance(case, dict):
                original_truncated = truncate_text(case.get('original', ''), truncated_length, 'token')
                watermarked_truncated = truncate_text(case.get('watermarked', ''), truncated_length, 'token')
                case_confidence = case.get('confidence', 0.5)
            else:
                continue

            prompt += f"Original: {original_truncated}\n"
            prompt += f"Watermarked: {watermarked_truncated}\n"
            prompt += f"(Confidence: {case_confidence:.2f})\n\n"

    prompt += instruction_2
    return prompt

def identify_watermark_effectiveness(original_text, watermarked_text, examples):
    """
    Determines if a watermarked text is effectively detected AND the original text is correctly classified.

    Args:
        original_text: The original unwatermarked text
        watermarked_text: The text with applied watermark
        examples: Previous examples of watermarked/original texts for few-shot learning

    Returns:
        Tuple (original_text, watermarked_text, label) where:
        - label = 'Good' 当且仅当：
            1. 水印文本被检测为 Watermarked
            2. 原始文本被检测为 Original
        - 否则为 'Bad'
    """
    # 转换示例格式
    detector_examples = convert_to_detector_examples(examples) if isinstance(examples, dict) else examples

    # 预处理水印文本，使用统一的前缀移除逻辑
    processed_watermarked = remove_common_prefixes(watermarked_text)

    # 检测水印文本
    prompt_watermarked = make_prompt_for_detection(processed_watermarked, detector_examples[:5])
    res_watermarked = call_model_api(prompt_watermarked)  # 调用模型API获取分类结果

    # 检测原始文本
    prompt_original = make_prompt_for_detection(original_text, detector_examples[:5])
    res_original = call_model_api(prompt_original)

    # 判断逻辑
    is_watermarked_detected = "watermarked" in res_watermarked.lower()
    is_original_correct = "original" in res_original.lower()

    # 完整记录检测结果，便于调试
    detection_result = {
        "watermarked_text_result": res_watermarked,
        "original_text_result": res_original,
        "is_watermarked_detected": is_watermarked_detected,
        "is_original_correct": is_original_correct
    }

    label = 'Good' if (is_watermarked_detected and is_original_correct) else 'Bad'
    return (res_original, res_watermarked, label)

def process_reply_from_qwen(res):
    if not res:
        return None
    res = res.lower()
    if 'original' in res and 'watermarked' not in res:
        return '0'
    elif 'watermarked' in res and 'original' not in res:
        return '1'
    else:
        return None  # 模糊不清的情况

def compute_three_recalls(labels, preds):
    all_n, all_p, tn, tp = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == '0':
            all_p += 1
        if label == '1':
            all_n += 1
        if label == pred == '0':
            tp += 1
        if label == pred == '1':
            tn += 1
    human_rec, machine_rec = tp * 100 / all_p, tn * 100 / all_n
    avg_rec = (human_rec + machine_rec) / 2
    return (human_rec, machine_rec, avg_rec)


def compute_metrics(labels, preds):
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    acc, precision, recall, f1 = accuracy_score(labels, preds), precision_score(labels, preds, pos_label='1'), recall_score(labels, preds, pos_label='1'), f1_score(labels, preds, pos_label='1')
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)

# 简单但改进的文本嵌入函数
def get_text_embedding(text, dimension=100):
    """
    生成文本的嵌入向量
    优先使用sentence-transformers，回退到改进的简单方法
    """
    if not text:
        return np.zeros(dimension)

    # 如果sentence-transformers可用，使用它
    if SENTENCE_TRANSFORMERS_AVAILABLE and sentence_model is not None:
        try:
            embedding = sentence_model.encode([text])[0]
            # 如果维度不匹配，进行调整
            if len(embedding) != dimension:
                if len(embedding) > dimension:
                    return embedding[:dimension]
                else:
                    # 扩展到所需维度
                    extended = np.zeros(dimension)
                    extended[:len(embedding)] = embedding
                    return extended
            return embedding
        except Exception as e:
            print(f"Warning: sentence-transformers encoding failed: {e}")
            # 继续使用简单方法

    # 改进的简单嵌入方法
    # 使用TF-IDF风格的方法而不是简单的词袋
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return np.zeros(dimension)

    # 计算词频
    word_counts = Counter(words)
    total_words = len(words)

    # 创建向量
    vector = np.zeros(dimension)

    for word, count in word_counts.items():
        # 使用词频和逆文档频率的近似
        tf = count / total_words

        # 为每个词生成多个特征
        for i in range(3):  # 每个词生成3个特征
            hash_val = (hash(word + str(i)) % dimension)
            vector[hash_val] += tf * (1.0 / (i + 1))  # 权重递减

    # 添加一些n-gram特征
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i + 1]
        hash_val = hash(bigram) % dimension
        vector[hash_val] += 0.5  # bigram权重

    # 归一化
    norm_val = np.linalg.norm(vector)
    if norm_val > 0:
        vector = vector / norm_val

    return vector

# 嵌入缓存
embedding_cache = {}

def get_cached_embedding(text, dimension=100):
    """缓存版本的文本嵌入获取"""
    if text not in embedding_cache:
        embedding_cache[text] = get_text_embedding(text, dimension)
    return embedding_cache[text]

def get_comprehensive_similarity(text1, text2, weights=None):
    """
    计算两个文本的综合相似度
    使用多维度评估：词汇重叠、语义相似度、结构相似度等

    :param text1: 文本1
    :param text2: 文本2
    :param weights: 权重字典，可选
    :return: 综合相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # 默认权重
    if weights is None:
        weights = {
            'lexical': 0.3,    # 词汇相似度
            'semantic': 0.4,   # 语义相似度
            'structure': 0.2,  # 结构相似度
            'length': 0.1      # 长度相似度
        }

    try:
        # 1. 词汇相似度 (简单的词汇重叠)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            lexical_sim = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            lexical_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            lexical_sim = intersection / union if union > 0 else 0.0

        # 2. 语义相似度 (使用embedding)
        try:
            emb1 = get_cached_embedding(text1[:500])  # 限制长度避免过长
            emb2 = get_cached_embedding(text2[:500])

            # 计算余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                semantic_sim = 0.0
            else:
                semantic_sim = max(0.0, dot_product / (norm1 * norm2))
        except Exception:
            semantic_sim = lexical_sim  # 回退到词汇相似度

        # 3. 结构相似度 (句子数、平均句长等)
        sentences1 = text1.split('.')
        sentences2 = text2.split('.')

        sent_count_ratio = min(len(sentences1), len(sentences2)) / max(len(sentences1), len(sentences2), 1)

        avg_len1 = sum(len(s.split()) for s in sentences1) / max(len(sentences1), 1)
        avg_len2 = sum(len(s.split()) for s in sentences2) / max(len(sentences2), 1)
        avg_len_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2, 1)

        structure_sim = (sent_count_ratio + avg_len_ratio) / 2

        # 4. 长度相似度
        len1, len2 = len(text1), len(text2)
        length_sim = min(len1, len2) / max(len1, len2, 1)

        # 综合相似度
        comprehensive_sim = (
            weights['lexical'] * lexical_sim +
            weights['semantic'] * semantic_sim +
            weights['structure'] * structure_sim +
            weights['length'] * length_sim
        )

        return min(1.0, max(0.0, comprehensive_sim))

    except Exception as e:
        print(f"Warning: Error in comprehensive similarity calculation: {e}")
        # 回退到简单的词汇重叠
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0

def get_confidence_voting(text, detector_examples, rounds=5):
    """
    通过多次检测投票获取置信度
    :param text: 待检测文本
    :param detector_examples: 检测示例
    :param rounds: 检测轮次
    :return: 置信度 (0-1)
    """
    votes = 0

    # 确保文本已经去除前缀
    processed_text = remove_common_prefixes(text)

    # 确保有足够的示例进行采样
    if len(detector_examples) == 0:
        return 0.5  # 默认置信度

    for _ in range(rounds):
        # 随机选择不同的少量示例，增加多样性
        sample_size = min(len(detector_examples), 5)
        random_examples = random.sample(detector_examples, sample_size)
        prompt = make_prompt_for_detection(processed_text, random_examples)
        response = call_model_api(prompt).lower()
        if 'watermarked' in response:
            votes += 1
        time.sleep(0.5)  # 控制请求频率

    # 计算置信度
    confidence = votes / rounds
    return confidence

class EntropyBasedConfidence:
    """基于信息熵的置信度计算"""

    def __init__(self, min_rounds=3, max_rounds=7, confidence_threshold=0.8):
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold

    def calculate_entropy(self, predictions):
        """计算预测分布的熵"""
        if not predictions:
            return float('inf')

        counter = Counter(predictions)
        total = len(predictions)
        entropy = 0

        for count in counter.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def adaptive_confidence_voting(self, text, detector_examples, detection_type="watermark"):
        """自适应置信度投票"""
        predictions = []
        confidences = []

        for round_num in range(self.max_rounds):
            # 执行检测
            prediction, individual_conf = self._single_detection_with_confidence(
                text, detector_examples, round_num
            )

            predictions.append(prediction)
            confidences.append(individual_conf)

            # 在最小轮数后开始检查是否可以提前停止
            if round_num >= self.min_rounds - 1:
                entropy = self.calculate_entropy(predictions)
                consistency = self._calculate_consistency(predictions)

                # 如果达到高一致性或低熵，可以提前停止
                if consistency >= self.confidence_threshold or entropy < 0.5:
                    break

        # 计算最终置信度
        final_confidence = self._calculate_weighted_confidence(predictions, confidences)
        final_prediction = self._get_weighted_prediction(predictions, confidences)

        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'rounds_used': len(predictions),
            'entropy': self.calculate_entropy(predictions),
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }

    def _single_detection_with_confidence(self, text, examples, round_num):
        """单次检测并提取置信度信息"""
        # 构造带置信度请求的提示
        confidence_prompt = self._make_confidence_prompt(text, examples, round_num)

        try:
            response = call_model_api(confidence_prompt)
            prediction, confidence = self._parse_confidence_response(response)
            return prediction, confidence
        except:
            # 回退到简单检测
            simple_prompt = make_prompt_for_detection(text, examples)
            response = call_model_api(simple_prompt)
            prediction = response.lower()
            return prediction, 0.5  # 默认置信度

    def _make_confidence_prompt(self, text, examples, round_num):
        """构造包含置信度评估的提示"""
        base_prompt = make_prompt_for_detection(text, examples)

        confidence_instruction = """

在给出检测结果后，请评估你的置信度(0-1之间的数值)：
- 1.0: 非常确定
- 0.8: 比较确定
- 0.6: 中等确定
- 0.4: 不太确定
- 0.2: 很不确定

请按以下格式回答：
预测: [Original/Watermarked]
置信度: [0.0-1.0的数值]
推理: [简短说明判断依据]
"""

        return base_prompt + confidence_instruction

    def _parse_confidence_response(self, response):
        """解析包含置信度的响应"""
        import re

        # 提取预测
        pred_match = re.search(r'预测\s*:\s*(Original|Watermarked)', response, re.IGNORECASE)
        prediction = pred_match.group(1).lower() if pred_match else "original"

        # 提取置信度
        conf_match = re.search(r'置信度\s*:\s*([0-9]*\.?[0-9]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        return prediction, max(0.0, min(1.0, confidence))

    def _calculate_consistency(self, predictions):
        """计算预测一致性"""
        if not predictions:
            return 0.0

        counter = Counter(predictions)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(predictions)

    def _calculate_weighted_confidence(self, predictions, confidences):
        """计算加权置信度"""
        if not predictions:
            return 0.0

        # 基于个体置信度的加权平均
        weighted_sum = sum(conf for conf in confidences)
        total_weight = len(confidences)

        base_confidence = weighted_sum / total_weight

        # 根据一致性调整
        consistency = self._calculate_consistency(predictions)
        consistency_bonus = consistency * 0.2

        return min(1.0, base_confidence + consistency_bonus)

    def _get_weighted_prediction(self, predictions, confidences):
        """基于置信度加权的最终预测"""
        prediction_scores = {}

        for pred, conf in zip(predictions, confidences):
            if pred not in prediction_scores:
                prediction_scores[pred] = 0
            prediction_scores[pred] += conf

        return max(prediction_scores.items(), key=lambda x: x[1])[0]

def find_similar_cases(query_text, cases, top_k=10):
    """
    基于语义相似度查找最相似的案例
    :param query_text: 查询文本
    :param cases: 案例列表，每个案例可以是EnhancedCase对象或包含'original'和'watermarked'的字典
    :param top_k: 返回的最相似案例数量
    :return: 相似案例列表
    """
    if not cases or len(cases) == 0:
        return []

    # 获取查询文本的嵌入
    query_embed = get_cached_embedding(query_text).reshape(1, -1)

    # 获取所有案例原始文本的嵌入
    corpus_embeds = []
    valid_cases = []

    for case in cases:
        try:
            # 兼容不同的数据结构
            if hasattr(case, 'original'):
                # 如果是EnhancedCase对象
                case_text = case.original
            elif isinstance(case, dict) and 'original' in case:
                # 如果是字典
                case_text = case['original']
            else:
                # 防御性编程，处理可能的异常数据结构
                continue

            if case_text:
                case_embed = get_cached_embedding(case_text)
                corpus_embeds.append(case_embed)
                valid_cases.append(case)
        except Exception as e:
            print(f"处理案例时出错: {e}")
            continue

    if not corpus_embeds:
        return []

    # 转换为numpy数组
    corpus_embeds = np.array(corpus_embeds)

    # 计算相似度
    similarities = cosine_similarity(query_embed, corpus_embeds)[0]

    # 获取相似度最高的索引
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # 返回相似度最高的案例
    return [valid_cases[i] for i in top_indices if i < len(valid_cases)]

class DifficultyAwareConfidence:
    """基于检测难度的置信度系统"""

    def __init__(self):
        self.difficulty_indicators = {
            'length_similarity': 0.2,
            'semantic_similarity': 0.3,
            'style_consistency': 0.2,
            'example_diversity': 0.3
        }

    def assess_detection_difficulty(self, text, examples):
        """评估检测难度"""
        difficulty_score = 0

        # 1. 长度相似度 - 长度太相似会增加难度
        if examples["success"]:
            avg_length_ratio = np.mean([
                min(len(text), len(case.watermarked)) / max(len(text), len(case.watermarked))
                for case in examples["success"][:5]
            ])
            length_difficulty = avg_length_ratio  # 越相似越难检测
            difficulty_score += self.difficulty_indicators['length_similarity'] * length_difficulty

        # 2. 语义相似度 - 语义太相似增加难度
        if examples["success"]:
            semantic_similarities = [
                get_comprehensive_similarity(text, case.watermarked)
                for case in examples["success"][:5]
            ]
            avg_semantic_sim = np.mean(semantic_similarities)
            semantic_difficulty = avg_semantic_sim  # 越相似越难
            difficulty_score += self.difficulty_indicators['semantic_similarity'] * semantic_difficulty

        # 3. 示例多样性 - 示例不够多样化增加难度
        example_diversity = self._calculate_example_diversity(examples)
        diversity_difficulty = 1.0 - example_diversity  # 多样性低则难度高
        difficulty_score += self.difficulty_indicators['example_diversity'] * diversity_difficulty

        return min(1.0, difficulty_score)

    def adaptive_rounds_by_difficulty(self, difficulty_score):
        """根据难度自适应确定检测轮数"""
        if difficulty_score < 0.3:
            return 3  # 简单案例，3轮足够
        elif difficulty_score < 0.7:
            return 5  # 中等难度，5轮
        else:
            return 7  # 困难案例，7轮

    def difficulty_aware_confidence(self, text, examples, detection_type="watermark"):
        """基于难度感知的置信度计算"""
        # 1. 评估检测难度
        difficulty = self.assess_detection_difficulty(text, examples)

        # 2. 根据难度确定检测策略
        rounds_needed = self.adaptive_rounds_by_difficulty(difficulty)

        # 3. 执行自适应检测
        entropy_system = EntropyBasedConfidence(
            min_rounds=max(2, rounds_needed-2),
            max_rounds=rounds_needed
        )

        result = entropy_system.adaptive_confidence_voting(text, examples, detection_type)

        # 4. 根据难度调整最终置信度
        difficulty_penalty = difficulty * 0.3  # 难度越高，置信度越需要折扣
        adjusted_confidence = result['confidence'] * (1.0 - difficulty_penalty)

        result['confidence'] = max(0.1, adjusted_confidence)
        result['difficulty_score'] = difficulty
        result['difficulty_adjusted'] = True

        return result

    def _calculate_example_diversity(self, examples):
        """计算示例多样性"""
        if not examples["success"] or len(examples["success"]) < 2:
            return 0.0

        # 计算示例之间的平均相似度，相似度越低多样性越高
        similarities = []
        success_cases = examples["success"][:10]  # 只考虑前10个案例

        for i in range(len(success_cases)):
            for j in range(i+1, len(success_cases)):
                sim = get_comprehensive_similarity(
                    success_cases[i].original,
                    success_cases[j].original
                )
                similarities.append(sim)

        if not similarities:
            return 0.0

        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity  # 相似度低则多样性高

        return max(0.0, min(1.0, diversity))

class ImprovedConfidenceSystem:
    """改进的综合置信度系统"""

    def __init__(self, method="adaptive", **kwargs):
        self.method = method

        if method == "entropy":
            self.confidence_calculator = EntropyBasedConfidence(**kwargs)
        elif method == "difficulty":
            self.confidence_calculator = DifficultyAwareConfidence()
        elif method == "adaptive":
            self.entropy_calc = EntropyBasedConfidence(**kwargs)
            self.difficulty_calc = DifficultyAwareConfidence()
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_confidence(self, text, examples, detection_type="watermark"):
        """统一的置信度计算接口"""

        if self.method == "entropy":
            return self.confidence_calculator.adaptive_confidence_voting(text, examples, detection_type)

        elif self.method == "difficulty":
            return self.confidence_calculator.difficulty_aware_confidence(text, examples, detection_type)

        elif self.method == "adaptive":
            # 组合两种方法
            difficulty_result = self.difficulty_calc.difficulty_aware_confidence(text, examples, detection_type)

            # 如果难度很高，使用更保守的熵方法
            if difficulty_result['difficulty_score'] > 0.7:
                entropy_result = self.entropy_calc.adaptive_confidence_voting(text, examples, detection_type)

                # 取更保守的结果
                final_confidence = min(difficulty_result['confidence'], entropy_result['confidence'])
                final_prediction = difficulty_result['prediction']  # 优先使用难度感知的预测

                return {
                    'prediction': final_prediction,
                    'confidence': final_confidence,
                    'method_used': 'adaptive_conservative',
                    'difficulty_score': difficulty_result['difficulty_score'],
                    'entropy_result': entropy_result,
                    'difficulty_result': difficulty_result
                }
            else:
                return difficulty_result

