# KuaiRand Variable Dictionary + Basic Stats

Official source (field definitions): https://github.com/Kuairand/KuaiRand (README -> Data Descriptions)

Local stats source: your current files in `Data/data` (or `data/data`)

## log

| column | official_type | local_dtype | row_count | unique_count | na_count | na_rate | min_value | max_value | mean_value | official_description | sample_values |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|comment_stay_time|int64|int64|2622668|24974|0|0.000000|0|300000|327.268|Time the user stayed in the comments section.|0 ;  0 ;  0|
|date|int64|int64|2622668|30|0|0.000000|2.02204e+07|2.02205e+07|2.02204e+07|The date of this interaction.|20220411 ;  20220416 ;  20220420|
|duration_ms|int64|int64|2622668|5757|0|0.000000|0|1.17772e+06|101688|The video duration in milliseconds.|209900 ;  65400 ;  170833|
|hourmin|int64|int64|2622668|24|0|0.000000|0|2300|1430.75|The time of this interaction (format: HHSS).|1900 ;  2000 ;  1600|
|is_click|int64|int64|2622668|2|0|0.000000|0|1|0.331447|Binary feedback. In two-column UI: click; in single-column UI: valid_play rule based on play_time_ms and duration_ms.|0 ;  0 ;  0|
|is_comment|int64|int64|2622668|2|0|0.000000|0|1|0.00155109|Binary feedback indicating the user wrote a comment.|0 ;  0 ;  0|
|is_follow|int64|int64|2622668|2|0|0.000000|0|1|0.000706532|Binary feedback indicating the user hit the follow-the-author button.|0 ;  0 ;  0|
|is_forward|int64|int64|2622668|2|0|0.000000|0|1|0.000680223|Binary feedback indicating the user forwarded this video.|0 ;  0 ;  0|
|is_hate|int64|int64|2622668|2|0|0.000000|0|1|0.000785841|Binary feedback indicating the user hated this video.|0 ;  0 ;  0|
|is_like|int64|int64|2622668|2|0|0.000000|0|1|0.0122917|Binary feedback indicating the user hit the like button.|0 ;  0 ;  0|
|is_profile_enter|int64|int64|2622668|2|0|0.000000|0|1|0.0156684|Binary feedback indicating the user enters the author profile.|0 ;  0 ;  0|
|is_rand|int64|int64|2622668|2|0|0.000000|0|1|0.452234|Binary indicator whether this is random intervention exposure.|0 ;  0 ;  0|
|long_view|int64|int64|2622668|2|0|0.000000|0|1|0.220193|Binary feedback for long-time play based on play_time_ms and duration_ms threshold rules.|0 ;  0 ;  0|
|play_time_ms|int64|int64|2622668|160602|0|0.000000|0|1.02381e+06|15676.5|The user's view time in milliseconds.|1385 ;  0 ;  1405|
|profile_stay_time|int64|int64|2622668|184|0|0.000000|0|300000|1.88195|Time the user stayed on the author profile.|0 ;  0 ;  0|
|tab|int64|int64|2622668|15|0|0.000000|0|14|1.20107|Scenario indicator of this interaction (range [0,14]).|1 ;  0 ;  1|
|time_ms|int64|int64|2622668|2512814|0|0.000000|1.64948e+12|1.65203e+12|1.65072e+12|The timestamp of this interaction in milliseconds.|1649675512388 ;  1650111976017 ;  1650444367095|
|user_id|int64|int64|2622668|27285|0|0.000000|0|27284|13610.8|The ID of the video.|0 ;  0 ;  0|
|video_id|int64|int64|2622668|7583|0|0.000000|0|7582|3799.63|The ID of the video.|1527 ;  7405 ;  6026|

## user_features

| column | official_type | local_dtype | row_count | unique_count | na_count | na_rate | min_value | max_value | mean_value | official_description | sample_values |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|fans_user_num|int64|int64|27285|2210|0|0.000000|0|1.98625e+06|602.728|Number of fans of this user.|150 ;  20 ;  26|
|fans_user_num_range|str|str|27285|9|0|0.000000||||Range bucket of fans_user_num.|[100,1k) ;  [10,100) ;  [10,100)|
|follow_user_num|int64|int64|27285|2562|0|0.000000|0|5003|409.833|Number of users that this user follows.|514 ;  457 ;  8|
|follow_user_num_range|str|str|27285|8|0|0.000000||||Range bucket of follow_user_num.|500+ ;  (250,500] ;  (0,10]|
|friend_user_num|int64|int64|27285|1391|0|0.000000|0|4995|119.298|Number of friends this user has.|34 ;  3 ;  3|
|friend_user_num_range|str|str|27285|7|0|0.000000||||Range bucket of friend_user_num.|[30,60) ;  [1,5) ;  [1,5)|
|is_live_streamer|int64|int64|27285|2|0|0.000000|-124|1|-95.7885|Whether this user is a live streamer.|1 ;  -124 ;  -124|
|is_lowactive_period|int64|int64|27285|1|0|0.000000|0|0|0|Whether this user is in low-active period.|0 ;  0 ;  0|
|is_video_author|int64|int64|27285|2|0|0.000000|0|1|0.816419|Whether this user has uploaded videos.|1 ;  1 ;  1|
|onehot_feat0|int64|int64|27285|2|0|0.000000|0|1|0.563093|Encrypted one-hot feature. Position index of 1.|1 ;  1 ;  1|
|onehot_feat1|int64|int64|27285|7|0|0.000000|0|6|2.95807|Encrypted feature.|1 ;  3 ;  0|
|onehot_feat10|int64|int64|27285|5|0|0.000000|0|4|2.38043|Encrypted feature.|3 ;  2 ;  2|
|onehot_feat11|int64|int64|27285|5|0|0.000000|0|4|0.500055|Encrypted feature.|0 ;  2 ;  2|
|onehot_feat12|int64|float64|27285|2|714|0.026168|0|1|0.723872|Encrypted feature.|0.0 ;  1.0 ;  0.0|
|onehot_feat13|int64|float64|27285|2|714|0.026168|0|1|0.0681946|Encrypted feature.|1.0 ;  0.0 ;  0.0|
|onehot_feat14|int64|float64|27285|2|714|0.026168|0|1|0.119115|Encrypted feature.|0.0 ;  1.0 ;  0.0|
|onehot_feat15|int64|float64|27285|2|714|0.026168|0|1|0.0226939|Encrypted feature.|0.0 ;  0.0 ;  1.0|
|onehot_feat16|int64|float64|27285|2|714|0.026168|0|1|0.0403824|Encrypted feature.|0.0 ;  0.0 ;  0.0|
|onehot_feat17|int64|float64|27285|2|714|0.026168|0|1|0.0144895|Encrypted feature.|0.0 ;  0.0 ;  0.0|
|onehot_feat2|int64|int64|27285|50|0|0.000000|0|49|16.2174|Encrypted feature.|29 ;  2 ;  2|
|onehot_feat3|int64|int64|27285|1471|0|0.000000|0|1470|784.658|Encrypted feature.|949 ;  1160 ;  1176|
|onehot_feat4|int64|float64|27285|15|874|0.032032|0|14|2.16501|Encrypted feature.|1.0 ;  4.0 ;  2.0|
|onehot_feat5|int64|int64|27285|34|0|0.000000|0|33|0.0794942|Encrypted feature.|0 ;  0 ;  0|
|onehot_feat6|int64|int64|27285|3|0|0.000000|0|2|0.465934|Encrypted feature.|0 ;  0 ;  0|
|onehot_feat7|int64|int64|27285|118|0|0.000000|0|117|21.0366|Encrypted feature.|14 ;  31 ;  31|
|onehot_feat8|int64|int64|27285|454|0|0.000000|0|453|231.789|Encrypted feature.|135 ;  283 ;  275|
|onehot_feat9|int64|int64|27285|7|0|0.000000|0|6|3.5536|Encrypted feature.|6 ;  6 ;  5|
|register_days|int64|int64|27285|2813|0|0.000000|14|3624|1204.4|Days since user registration.|799 ;  1474 ;  231|
|register_days_range|str|str|27285|8|0|0.000000||||Range bucket of register_days.|730+ ;  730+ ;  181-365|
|user_active_degree|str|str|27285|9|0|0.000000||||In {'high_active','full_active','middle_active','UNKNOWN'}.|full_active ;  full_active ;  full_active|
|user_id|int64|int64|27285|27285|0|0.000000|0|27284|13642|The ID of the user.|0 ;  1 ;  2|

## video_features_basic

| column | official_type | local_dtype | row_count | unique_count | na_count | na_rate | min_value | max_value | mean_value | official_description | sample_values |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|author_id|int64|int64|7583|6510|0|0.000000|0|8.73398e+06|5.1373e+06|The ID of the author of this video.|7349781 ;  2103883 ;  5067285|
|music_id|int64|int64|7583|7202|0|0.000000|0|9.17022e+09|8.51053e+09|Background music ID.|9155697141 ;  6355810746 ;  6618412736|
|music_type|int64|float64|7583|5|203|0.026770|4|11|8.54634|Background music type.|9.0 ;  9.0 ;  4.0|
|server_height|int64|float64|7583|120|0|0.000000|448|2400|1140.16|Video height on server.|1280.0 ;  1280.0 ;  1280.0|
|server_width|int64|float64|7583|156|0|0.000000|270|2400|850.03|Video width on server.|720.0 ;  720.0 ;  720.0|
|tag|str|str|7583|110|96|0.012660||||List of key categories/tags of this video.|39 ;  2 ;  1|
|upload_dt|str|str|7583|3|0|0.000000||||Upload date of this video.|2022-04-10 ;  2022-04-10 ;  2022-04-09|
|upload_type|str|str|7583|14|0|0.000000||||Upload type of this video.|LongImport ;  Kmovie ;  ShortImport|
|video_duration|int64|float64|7583|5756|239|0.031518|5000|1.17772e+06|108616|Video duration in milliseconds.|87433.0 ;  218066.0 ;  9233.0|
|video_id|int64|int64|7583|7583|0|0.000000|0|7582|3791|The ID of the video.|0 ;  1 ;  2|
|video_type|str|str|7583|3|0|0.000000||||Type of this video (NORMAL or AD).|NORMAL ;  NORMAL ;  NORMAL|
|visible_status|int|float64|7583|1|0|0.000000|0|0|0|Current visible status in app.|0.0 ;  0.0 ;  0.0|

## video_features_statistic

| column | official_type | local_dtype | row_count | unique_count | na_count | na_rate | min_value | max_value | mean_value | official_description | sample_values |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|cancel_collect_cnt|float64|float64|7583|6680|0|0.000000|0|771.607|7.61989|Average remove-from-favorites actions.|0.2352941176470588 ;  1.2794117647058822 ;  0.6813186813186813|
|cancel_collect_user_num|float64|float64|7583|6599|0|0.000000|0|749.113|7.16001|Average users removing from favorites.|0.2352941176470588 ;  1.1764705882352942 ;  0.6593406593406593|
|cancel_follow_cnt|float64|float64|7583|5999|0|0.000000|0|127.277|2.31941|Average decreased follows from this video.|0.5196078431372549 ;  0.5735294117647058 ;  0.0769230769230769|
|cancel_follow_user_num|float64|float64|7583|5968|0|0.000000|0|127.019|2.29811|Average users canceling follow due to this video.|0.5098039215686274 ;  0.5735294117647058 ;  0.0769230769230769|
|cancel_like_cnt|float64|float64|7583|7136|0|0.000000|0.025974|1443.67|18.7562|Average canceled likes.|0.696078431372549 ;  3.419117647058824 ;  0.3736263736263736|
|cancel_like_user_num|float64|float64|7583|7107|0|0.000000|0.025974|1160.36|16.3591|Average users canceling likes.|0.6568627450980392 ;  2.8529411764705883 ;  0.3516483516483517|
|click_like_cnt|float64|float64|7583|7488|0|0.000000|0.237624|7188.6|104.859|Average likes from click-like behavior.|2.92156862745098 ;  22.23529411764705 ;  5.175824175824176|
|collect_cnt|float64|float64|7583|7290|0|0.000000|0.0294118|5866.92|28.7477|Average add-to-favorites actions.|0.9019607843137256 ;  6.6838235294117645 ;  1.835164835164835|
|collect_user_num|float64|float64|7583|7274|0|0.000000|0.0294118|5798.58|27.6245|Average users adding to favorites.|0.8823529411764706 ;  6.470588235294118 ;  1.7912087912087913|
|comment_cnt|float64|float64|7583|6906|0|0.000000|0|3116.98|12.9337|Average comments.|0.196078431372549 ;  2.794117647058824 ;  1.2967032967032968|
|comment_like_cnt|float64|float64|7583|6867|0|0.000000|0|8177.85|41.221|Average comment likes.|0.0392156862745098 ;  2.6911764705882355 ;  0.0659340659340659|
|comment_like_user_num|float64|float64|7583|6592|0|0.000000|0|2866.91|20.9706|Average users liking comments.|0.0392156862745098 ;  1.4779411764705883 ;  0.0659340659340659|
|comment_stay_duration|float64|float64|7583|7583|0|0.000000|922.716|1.53523e+09|6.40493e+06|Average total comment-stay duration.|203428.16666666663 ;  797810.2720588235 ;  54158.38461538462|
|comment_user_num|float64|float64|7583|6834|0|0.000000|0|2613.25|10.8925|Average users commenting.|0.196078431372549 ;  2.3455882352941178 ;  1.2087912087912087|
|complete_play_cnt|float64|float64|7583|7573|0|0.000000|1.10811|256630|1887.16|Average complete plays.|195.2549019607843 ;  130.50735294117646 ;  80.20879120879121|
|complete_play_user_num|float64|float64|7583|7572|0|0.000000|0.945946|241738|1811.51|Average users with complete play.|183.09803921568627 ;  125.94117647058825 ;  75.46153846153847|
|counts|int64|int64|7583|125|0|0.000000|45|181|141.837|Number of statistics records used for averaging.|102 ;  136 ;  91|
|delete_comment_cnt|float64|float64|7583|3635|0|0.000000|0|400.821|0.679197|Average deleted comments.|0.0196078431372549 ;  0.1029411764705882 ;  0.0|
|delete_comment_user_num|float64|float64|7583|3448|0|0.000000|0|181.26|0.533816|Average users deleting comments.|0.0196078431372549 ;  0.0955882352941176 ;  0.0|
|direct_comment_cnt|float64|float64|7583|6772|0|0.000000|0|2071.56|8.9109|Average direct comments (depth=1).|0.1568627450980392 ;  2.0073529411764706 ;  1.1868131868131868|
|direct_comment_user_num|float64|float64|7583|6771|0|0.000000|0|1368.11|8.15785|Average users writing direct comments.|0.1568627450980392 ;  1.8676470588235288 ;  1.1538461538461535|
|double_click_cnt|float64|float64|7583|7477|0|0.000000|0.194805|10136.7|123.885|Average double-click count.|3.519607843137255 ;  15.867647058823527 ;  1.4395604395604396|
|download_cnt|float64|float64|7583|5346|0|0.000000|0|639.568|4.42328|Average downloads.|0.0196078431372549 ;  4.0588235294117645 ;  0.1098901098901098|
|download_user_num|float64|float64|7583|5314|0|0.000000|0|474.194|3.89305|Average users downloading.|0.0196078431372549 ;  3.977941176470588 ;  0.1098901098901098|
|follow_cnt|float64|float64|7583|7117|0|0.000000|0|1829.87|17.4086|Average increased follows from this video.|0.8431372549019608 ;  3.5 ;  0.2527472527472527|
|follow_user_num|float64|float64|7583|7076|0|0.000000|0|1828.08|17.3372|Average users following author due to this video.|0.8235294117647058 ;  3.492647058823529 ;  0.2527472527472527|
|like_cnt|float64|float64|7583|7525|0|0.000000|0.454545|17396.2|230.745|Average likes.|6.470588235294118 ;  38.16176470588236 ;  6.626373626373627|
|like_user_num|float64|float64|7583|7545|0|0.000000|0.441558|17290|226.8|Average users who like.|6.392156862745098 ;  37.34558823529412 ;  6.582417582417582|
|long_time_play_cnt|float64|float64|7583|7582|0|0.000000|0|301136|3687.35|Average long-time plays (rule-based).|318.97058823529414 ;  549.6544117647059 ;  79.86813186813187|
|long_time_play_user_num|float64|float64|7583|7578|0|0.000000|0|278443|3522.55|Average users with long-time play.|299.3333333333333 ;  518.1029411764706 ;  75.16483516483517|
|outsite_share_all_cnt|float64|float64|7583|6270|0|0.000000|0|1589.76|5.20573|Average shares outside Kuaishou app.|0.0784313725490196 ;  4.691176470588236 ;  0.2857142857142857|
|play_cnt|float64|float64|7583|7580|0|0.000000|33.3108|538564|7747.17|Average play count.|816.8823529411765 ;  2116.25 ;  425.68131868131866|
|play_duration|float64|float64|7583|7583|0|0.000000|191288|3.4713e+10|3.06803e+08|Average total play duration (ms).|27679151.656862747 ;  85921520.70588236 ;  4197048.472527472|
|play_progress|float64|float64|7583|7583|0|0.000000|0|0.571328|0.170996|Average play ratio = play_duration / video_duration.|0.1174066622659254 ;  0.0647258562331143 ;  0.2383424967385748|
|play_user_num|float64|float64|7583|7583|0|0.000000|29.9189|480039|6907.64|Average number of users who played.|713.1764705882352 ;  1864.0441176470588 ;  386.5384615384616|
|reduce_similar_cnt|float64|float64|7583|6802|0|0.000000|0.0263158|333.782|6.09569|Average reduce-similar actions.|0.9607843137254902 ;  2.764705882352941 ;  2.7142857142857144|
|reduce_similar_user_num|float64|float64|7583|6761|0|0.000000|0.0263158|286.644|5.76457|Average users choosing reduce-similar.|0.9117647058823528 ;  2.2794117647058822 ;  2.67032967032967|
|reply_comment_cnt|float64|float64|7583|5526|0|0.000000|0|1713.91|4.02276|Average reply comments (depth>1).|0.0392156862745098 ;  0.7867647058823529 ;  0.1098901098901098|
|reply_comment_user_num|float64|float64|7583|5192|0|0.000000|0|1459.99|3.211|Average users replying to comments.|0.0392156862745098 ;  0.5514705882352942 ;  0.0549450549450549|
|report_cnt|float64|float64|7583|406|0|0.000000|0|1.25373|0.00509813|Average reports.|0.0 ;  0.0 ;  0.0|
|report_user_num|float64|float64|7583|364|0|0.000000|0|1.14925|0.00447271|Average users reporting.|0.0 ;  0.0 ;  0.0|
|share_all_cnt|float64|float64|7583|6650|0|0.000000|0|1721.55|8.23323|Average all share attempts (success not required).|0.1176470588235294 ;  5.470588235294118 ;  0.2967032967032967|
|share_all_user_num|float64|float64|7583|6557|0|0.000000|0|1461.24|7.13748|Average users with share attempts.|0.1078431372549019 ;  4.963235294117647 ;  0.2747252747252747|
|share_cnt|float64|float64|7583|6064|0|0.000000|0|1354.12|4.91452|Average successful shares.|0.0588235294117647 ;  1.6985294117647058 ;  0.1978021978021978|
|share_user_num|float64|float64|7583|5983|0|0.000000|0|1151.01|4.40453|Average users with successful shares.|0.0588235294117647 ;  1.4779411764705883 ;  0.1758241758241758|
|short_time_play_cnt|float64|float64|7583|7582|0|0.000000|0|160785|2107.19|Average short-time plays.|227.09803921568627 ;  1071.0073529411766 ;  264.3956043956044|
|short_time_play_user_num|float64|float64|7583|7581|0|0.000000|0|144064|1918.93|Average users with short-time play.|206.37254901960785 ;  969.1470588235294 ;  245.52747252747253|
|show_cnt|float64|float64|7583|7583|0|0.000000|65.25|535130|10552.4|Average daily+scenario show count over one month.|2579.686274509804 ;  4027.992647058824 ;  666.7032967032967|
|show_user_num|float64|float64|7583|7583|0|0.000000|62.0441|472356|9252.09|Average daily+scenario unique users shown this video.|2308.323529411765 ;  3368.3676470588234 ;  547.2197802197802|
|valid_play_cnt|float64|float64|7583|7581|0|0.000000|5.77027|326530|4589.04|Average valid plays (rule-based).|443.0196078431373 ;  717.6764705882352 ;  95.72527472527472|
|valid_play_user_num|float64|float64|7583|7578|0|0.000000|5.24324|300770|4348.71|Average users with valid play.|414.4313725490196 ;  669.3161764705883 ;  90.04395604395604|
|video_id|int64|int64|7583|7583|0|0.000000|0|7582|3791|The ID of the video.|0 ;  1 ;  2|
