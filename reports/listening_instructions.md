# 匿名听评说明

正式音频生成结束后，向每位听评者单独复制 `outputs/listening_test/participant/` 文件夹。不要发送 `outputs/listening_test/private/answer_key.json`。每位听评者使用不同的匿名 `rater_id`，并各自填写一份 `ratings_template.csv`；收回后把三份记录合并为一张 CSV，列名保持不变。

每一行对应同一文字提示的左右两段约 10 秒音频。请使用耳机，在安静环境中按自己的需要重复播放。分别判断：

1. `alignment_choice`：哪一段更符合文字描述，包括可听事件及其先后关系？
2. `naturalness_choice`：哪一段整体更自然，明显噪声、失真或静音更少？

两列分别填写 `L`（左）、`R`（右）或 `T`（难分高下）。不要根据文件名推测模型；文件名已匿名化。若两段都很差，仍可选 `T`，并把具体问题记录在单独备注中。请尽量完成全部 100 对；不要互相讨论后再填写，避免听评者之间的意见传染。

外部挑战提示另放在 `outputs/listening_challenge/participant/`，用于记录多事件是否出现和顺序是否正确。它与主测试集分开汇总；仅有两项主问题的 CSV 不能替代事件覆盖和顺序的人工记录。

至少收齐三位真实听评者的独立结果后，再使用私有答案键运行 `python -m tango2_rdpo.evaluate`。无论结果是否支持改进方法，保留每位听评者的原始表、汇总值和失败样例。
