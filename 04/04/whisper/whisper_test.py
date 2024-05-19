import whisper
import zhconv

# 英文
model = whisper.load_model("tiny")
result = model.transcribe("data/train.wav")
print(result["text"])

# 中文
print("origin text: 观众朋友们晚上好晚上好欢迎收看本期的盗月社蠢货新闻")
result = model.transcribe("data/train2.wav", language="Chinese")
s1 = zhconv.convert(result["text"], 'zh-cn')
print(s1)
