from lerobot.envs.utils import preprocess_observation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait

from subset_dataset import SubsetStateActionDataset
from phase_shift_dataset import PhaseShiftedDataset, PhaseSetDataset

STATE_KEEP_NAMES = [
    "ee_x_l", "ee_y_l", "ee_z_l",
    "ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l",
    "force_x_l", "force_y_l", "force_z_l",
    "torque_x_l", "torque_y_l", "torque_z_l",
    "ee_x_r", "ee_y_r", "ee_z_r",
    "ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r",
    "force_x_r", "force_y_r", "force_z_r",
    "torque_x_r", "torque_y_r", "torque_z_r",
    "stage",
]

ACTION_KEEP_NAMES = [
    "ee_x_l", "ee_y_l", "ee_z_l",
    "ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l",
    "gripper_l",
    "ee_x_r", "ee_y_r", "ee_z_r",
    "ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r",
    "gripper_r",
]
# Load the dataset
repo_id = "leledeyuan/cable_task2"
dataset = LeRobotDataset(
    repo_id = repo_id,
)

print("length of dataset:", len(dataset))

print("episodes:", dataset.num_episodes)


# dataset = SubsetStateActionDataset(dataset, STATE_KEEP_NAMES, ACTION_KEEP_NAMES)
# first_data = dataset[0]
# print("first data before phase shift:", first_data["observation.state"])

# dataset = PhaseShiftedDataset(dataset, phase_offset=5, main_task=1)

# first_data = dataset[0]
# print("first data:", first_data["observation.state"])
# print("meta action shape:", dataset.meta.features["action"]["shape"])
# print("meta state  shape:", dataset.meta.features["observation.state"]["shape"])
# if "observation.state" in first_data:
#     print("first data observation.state shape:", first_data["observation.state"].shape)

# metadata = dataset.meta
# print("fetures:", metadata.features["observation.state"].items())

# stats = metadata.stats
# print("stats:", stats["observation.state"])

# for k,v in stats["observation.state"].items():
#     print(f"shape of {k}:", v.shape)

# # print the count items
# print("count items:", stats["observation.state"]["count"])

# from tqdm import tqdm
# from lerobot.datasets.lerobot_dataset import LeRobotDataset


# dataset = LeRobotDataset(repo_id="leledeyuan/mixed-tshirt")
i = 15000

# # for i in tqdm(range(len(dataset))):
# #     try:
# #         sample = dataset[i]
# #     except Exception as e:
# #         print("❌ bad sample index:", i)
# #         print("error:", e)
# #         break

s = dataset[i]
print(s.keys())
# print("wrench_history:", s["observation.state.wrench_history_l"])
# print("episode_index:", int(s["episode_index"]))
# print("frame_index:", int(s["frame_index"]))
# print("timestamp:", float(s["timestamp"]))
# print("task:", s.get("task", None))