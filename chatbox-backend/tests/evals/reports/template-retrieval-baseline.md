# 模板匹配质量基准 · 2026-10-11

34 个案例中通过 28 个；流程正确率 82.4%。

有明确目标模板的案例：Top-1 正确率 96.9%，Recall@3 100.0%。

使用真实 embedding 和数据库检索；没有生成正文，也没有让模型自己评判答案。私人模板两个案例修正测试 ID 后单独复测，报告汇总两次有效观测。测试数据均已回滚。

| 失败案例 | 期望模板 | 实际结果 | 第一名分数 |
| --- | --- | --- | --- |
| meeting_summary_en | default:meeting_summary | default:meeting_agenda | 0.653 |
| meeting_summary_zh | default:meeting_summary | reject | 0.615 |
| professional_email_zh | default:professional_email | reject | 0.609 |
| customer_support_reply_zh | default:customer_support_reply | reject | 0.625 |
| meeting_agenda_zh | default:meeting_agenda | reject | 0.643 |
| incident_report_en | default:incident_report | reject | 0.635 |

目前 5 个失败案例找到了正确第一名，但得分低于 0.65 被拒绝；1 个案例把会议纪要选成会议议程。私人模板优先、相近私人模板等待用户选择，以及两条无关需求拒绝均通过。

这些是已知场景的初始基准，不能代表所有真实用户需求。新增需求应先标注期望模板，再测当前检索结果。
