安装xr_teleoperation到egodex环境

```
# 实时可视化
conda activate egodex


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python egodex_brainco_tools/viser_brainco_hand_only_viewer.py   --hdf5 egodex_example/clean_cups/0.hdf5   --fps 30   --loop   --port 8080

# 转换数据  默认开启Y轴自动居中，解决了歪的问题
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python egodex_brainco_tools/export_egodex_brainco_loop.py   --hdf5 egodex_example/clean_cups/1.hdf5   --config egodex_brainco_tools/config/brainco_vector.yml   --output-dir egodex_example/clean_cups/0_brainco_loop   --loops 1


# 播放数据
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python egodex_brainco_tools/viser_brainco_loop_json_viewer.py   --json egodex_example/clean_cups/0_brainco_loop/recomputed_ee_fullbody.json   --loop   --port 8080



```