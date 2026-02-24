# KuaiRand Metadata Notes (from official repository)

- Official source: https://github.com/Kuairand/KuaiRand
- Log tables contain interaction events with labels such as `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, and `long_view`.
- Core log time fields: `date` (YYYYMMDD), `hourmin` (HHMM), and `time_ms` (Unix timestamp in milliseconds).
- User table contains profile/behavioral attributes and anonymized one-hot features (`onehot_feat0` ... `onehot_feat17`).
- Video basic table contains static metadata (`video_type`, `upload_type`, `video_duration`, resolution, music info, tag).
- Video statistics table contains aggregate behavior metrics (show/play/like/comment/follow/share/collect style counts and user counts).

These notes are intended as a practical summary for EDA and feature engineering.
