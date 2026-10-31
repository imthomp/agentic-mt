# L1 translation-condition experiment

High-resource pair: en-fr. Low-resource pair(s): en-sw.

## Mean score per condition per pair

| ('pair', '')   | ('condition', '')   |   ('cometkiwi', 'mean') |   ('cometkiwi', 'std') |   ('cometkiwi', 'count') |   ('comet', 'mean') |   ('comet', 'std') |   ('comet', 'count') |   ('chrf', 'mean') |   ('chrf', 'std') |   ('chrf', 'count') |
|:---------------|:--------------------|------------------------:|-----------------------:|-------------------------:|--------------------:|-------------------:|---------------------:|-------------------:|------------------:|--------------------:|
| en-fr          | l1_baseline         |                0.685071 |              0.217999  |                      100 |            0.716356 |           0.220283 |                  100 |            55.5689 |          25.4762  |                 100 |
| en-fr          | l1_cot              |                0.304767 |              0.130921  |                      100 |            0.570244 |           0.17416  |                  100 |            22.1866 |          12.1612  |                 100 |
| en-fr          | l1_cot_tool         |                0.333221 |              0.129945  |                      100 |            0.501769 |           0.166833 |                  100 |            20.5231 |          12.9883  |                 100 |
| en-fr          | l1_tool             |                0.577171 |              0.223272  |                      100 |            0.660335 |           0.175238 |                  100 |            41.9278 |          24.3248  |                 100 |
| en-sw          | l1_baseline         |                0.444763 |              0.229929  |                      100 |            0.565948 |           0.209595 |                  100 |            29.0676 |          23.4761  |                 100 |
| en-sw          | l1_cot              |                0.296991 |              0.0763836 |                      100 |            0.458009 |           0.142407 |                  100 |            12.2815 |           7.65119 |                 100 |
| en-sw          | l1_cot_tool         |                0.338095 |              0.120749  |                      100 |            0.47017  |           0.138112 |                  100 |            15.2933 |           7.92989 |                 100 |
| en-sw          | l1_tool             |                0.395282 |              0.167501  |                      100 |            0.55581  |           0.13091  |                  100 |            23.5564 |          15.1671  |                 100 |

## Paired bootstrap significance between conditions

| pair   | metric    | condition_a   | condition_b   |   n |   mean_diff |        ci_lo |        ci_hi |   p_value |
|:-------|:----------|:--------------|:--------------|----:|------------:|-------------:|-------------:|----------:|
| en-fr  | cometkiwi | l1_cot_tool   | l1_cot        | 100 |   0.0284545 |  -0.00236766 |   0.0569526  |     0.068 |
| en-fr  | cometkiwi | l1_cot_tool   | l1_tool       | 100 |  -0.24395   |  -0.287906   |  -0.196769   |     0     |
| en-fr  | cometkiwi | l1_cot        | l1_baseline   | 100 |  -0.380305  |  -0.42682    |  -0.332368   |     0     |
| en-fr  | cometkiwi | l1_tool       | l1_baseline   | 100 |  -0.1079    |  -0.159621   |  -0.0564781  |     0     |
| en-fr  | cometkiwi | l1_cot_tool   | l1_baseline   | 100 |  -0.35185   |  -0.396      |  -0.304908   |     0     |
| en-sw  | cometkiwi | l1_cot_tool   | l1_cot        | 100 |   0.0411039 |   0.0180276  |   0.0654652  |     0.002 |
| en-sw  | cometkiwi | l1_cot_tool   | l1_tool       | 100 |  -0.057187  |  -0.0929947  |  -0.022772   |     0.002 |
| en-sw  | cometkiwi | l1_cot        | l1_baseline   | 100 |  -0.147772  |  -0.189449   |  -0.102228   |     0     |
| en-sw  | cometkiwi | l1_tool       | l1_baseline   | 100 |  -0.0494812 |  -0.0893805  |  -0.0074728  |     0.028 |
| en-sw  | cometkiwi | l1_cot_tool   | l1_baseline   | 100 |  -0.106668  |  -0.152019   |  -0.0613537  |     0     |
| en-fr  | comet     | l1_cot_tool   | l1_cot        | 100 |  -0.068474  |  -0.102291   |  -0.0318485  |     0     |
| en-fr  | comet     | l1_cot_tool   | l1_tool       | 100 |  -0.158565  |  -0.206239   |  -0.108201   |     0     |
| en-fr  | comet     | l1_cot        | l1_baseline   | 100 |  -0.146113  |  -0.195857   |  -0.0956293  |     0     |
| en-fr  | comet     | l1_tool       | l1_baseline   | 100 |  -0.0560213 |  -0.104161   |  -0.00534509 |     0.03  |
| en-fr  | comet     | l1_cot_tool   | l1_baseline   | 100 |  -0.214587  |  -0.26533    |  -0.167005   |     0     |
| en-sw  | comet     | l1_cot_tool   | l1_cot        | 100 |   0.0121611 |  -0.0198362  |   0.044342   |     0.414 |
| en-sw  | comet     | l1_cot_tool   | l1_tool       | 100 |  -0.0856392 |  -0.118433   |  -0.0544317  |     0     |
| en-sw  | comet     | l1_cot        | l1_baseline   | 100 |  -0.107938  |  -0.146907   |  -0.0680211  |     0     |
| en-sw  | comet     | l1_tool       | l1_baseline   | 100 |  -0.0101381 |  -0.0484966  |   0.026604   |     0.606 |
| en-sw  | comet     | l1_cot_tool   | l1_baseline   | 100 |  -0.0957773 |  -0.136572   |  -0.0496505  |     0     |
| en-fr  | chrf      | l1_cot_tool   | l1_cot        | 100 |  -1.66347   |  -4.24777    |   1.17279    |     0.234 |
| en-fr  | chrf      | l1_cot_tool   | l1_tool       | 100 | -21.4047    | -27.1149     | -16.0193     |     0     |
| en-fr  | chrf      | l1_cot        | l1_baseline   | 100 | -33.3823    | -39.0093     | -28.2878     |     0     |
| en-fr  | chrf      | l1_tool       | l1_baseline   | 100 | -13.6411    | -19.7431     |  -7.77647    |     0     |
| en-fr  | chrf      | l1_cot_tool   | l1_baseline   | 100 | -35.0458    | -40.5316     | -29.8833     |     0     |
| en-sw  | chrf      | l1_cot_tool   | l1_cot        | 100 |   3.01177   |   1.20878    |   4.84724    |     0.004 |
| en-sw  | chrf      | l1_cot_tool   | l1_tool       | 100 |  -8.26314   | -11.2869     |  -5.48401    |     0     |
| en-sw  | chrf      | l1_cot        | l1_baseline   | 100 | -16.786     | -21.6268     | -12.2307     |     0     |
| en-sw  | chrf      | l1_tool       | l1_baseline   | 100 |  -5.51114   |  -9.44706    |  -1.5385     |     0.002 |
| en-sw  | chrf      | l1_cot_tool   | l1_baseline   | 100 | -13.7743    | -18.194      |  -9.6463     |     0     |

## Core hypothesis test: does the condition-4 gap differ by resource level?

| metric    | gap                   | high_resource_pair   | low_resource_pair   |   gap_high_resource |   gap_low_resource |   diff_of_gaps |   diff_ci_lo |   diff_ci_hi |   p_value |
|:----------|:----------------------|:---------------------|:--------------------|--------------------:|-------------------:|---------------:|-------------:|-------------:|----------:|
| cometkiwi | l1_cot_tool - l1_cot  | en-fr                | en-sw               |           0.0288002 |          0.04063   |     -0.0118297 |   -0.0522627 |    0.0244711 |     0.56  |
| cometkiwi | l1_cot_tool - l1_tool | en-fr                | en-sw               |          -0.243805  |         -0.0568229 |     -0.186982  |   -0.241346  |   -0.135718  |     0     |
| comet     | l1_cot_tool - l1_cot  | en-fr                | en-sw               |          -0.0676145 |          0.0106837 |     -0.0782983 |   -0.126992  |   -0.0311865 |     0.002 |
| comet     | l1_cot_tool - l1_tool | en-fr                | en-sw               |          -0.158154  |         -0.0851185 |     -0.0730359 |   -0.129127  |   -0.017569  |     0.01  |
| chrf      | l1_cot_tool - l1_cot  | en-fr                | en-sw               |          -1.58824   |          2.96739   |     -4.55563   |   -7.67776   |   -1.46665   |     0.004 |
| chrf      | l1_cot_tool - l1_tool | en-fr                | en-sw               |         -21.3482    |         -8.17928   |    -13.1689    |  -19.7695    |   -7.17591   |     0     |
