"""Phase 3：TKE GPU 节点池的创建 / 删除（只读探测已在 plan.md 记录完毕后，
本文件是真正会花钱的一步——创建一个 GPU 节点，按量计费。

⚠️ 使用前确认：
  - 集群 / VPC / 子网 / 安全组 / 镜像等资源 ID 全部从环境变量读取，
    在 `.env` 中配置（`TKE_CLUSTER_ID` / `TKE_VPC_ID` / `TKE_SUBNET_ID` /
    `TKE_SECURITY_GROUP_ID` / `TKE_IMAGE_ID` / `TKE_REGION` / `TKE_ZONE`）
  - 只建 1 个节点，不开自动伸缩（EnableAutoscale=False，MinSize=MaxSize=DesiredCapacity=1）
  - 用完立刻调用 `delete` 释放，不要让它空跑

用法：
    python3 scripts/manage_gpu_nodepool.py create   # 创建节点池，等待节点 Ready
    python3 scripts/manage_gpu_nodepool.py status   # 查看节点池/节点状态
    python3 scripts/manage_gpu_nodepool.py delete   # 删除节点池（释放计费）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLUSTER_ID = os.environ.get("TKE_CLUSTER_ID", "YOUR_CLUSTER_ID")
REGION = os.environ.get("TKE_REGION", "ap-shanghai")
ZONE = os.environ.get("TKE_ZONE", "YOUR_ZONE")
VPC_ID = os.environ.get("TKE_VPC_ID", "YOUR_VPC_ID")
SUBNET_ID = os.environ.get("TKE_SUBNET_ID", "YOUR_SUBNET_ID")
SECURITY_GROUP_ID = os.environ.get("TKE_SECURITY_GROUP_ID", "YOUR_SG_ID")
INSTANCE_TYPE = "GN6S.LARGE20"
IMAGE_ID = os.environ.get("TKE_IMAGE_ID", "YOUR_IMAGE_ID")
NODE_POOL_OS = "ubuntu22.04x86_64"
NODE_POOL_NAME = "gpu-t4-pool"
STATE_FILE = ROOT / ".gpu_nodepool_state.json"


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def _tke_client():
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tke.v20180525 import tke_client

    sid = os.environ["TENCENTCLOUD_SECRET_ID"]
    skey = os.environ["TENCENTCLOUD_SECRET_KEY"]
    cred = credential.Credential(sid, skey)
    return tke_client.TkeClient(cred, REGION, ClientProfile(httpProfile=HttpProfile(reqTimeout=30)))


def cmd_create(args: argparse.Namespace) -> int:
    from tencentcloud.tke.v20180525 import models

    cli = _tke_client()

    as_para = {
        "MaxSize": 1,
        "MinSize": 1,
        "DesiredCapacity": 1,
        "VpcId": VPC_ID,
        "SubnetIds": [SUBNET_ID],
    }
    launch_para = {
        "InstanceType": INSTANCE_TYPE,
        "SystemDisk": {"DiskType": "CLOUD_PREMIUM", "DiskSize": 100},
        "InternetAccessible": {"PublicIpAssigned": False},
        "LoginSettings": {"Password": args.node_password},
        "SecurityGroupIds": [SECURITY_GROUP_ID],
        "InstanceChargeType": "POSTPAID_BY_HOUR",
    }

    req = models.CreateClusterNodePoolRequest()
    req.ClusterId = CLUSTER_ID
    req.Name = NODE_POOL_NAME
    req.AutoScalingGroupPara = json.dumps(as_para)
    req.LaunchConfigurePara = json.dumps(launch_para)
    req.EnableAutoscale = False
    req.NodePoolOs = NODE_POOL_OS
    req.OsCustomizeType = "GENERAL"

    adv = models.InstanceAdvancedSettings()
    adv.GPUArgs = models.GPUArgs()  # 空对象 = 让 TKE 用默认策略处理 GPU 驱动安装
    req.InstanceAdvancedSettings = adv

    rsp = cli.CreateClusterNodePool(req)
    node_pool_id = rsp.NodePoolId
    print(f"节点池已创建：NodePoolId={node_pool_id}")

    STATE_FILE.write_text(json.dumps({
        "node_pool_id": node_pool_id, "cluster_id": CLUSTER_ID, "created_at": time.time(),
    }), encoding="utf-8")
    print(f"状态已存到 {STATE_FILE}（delete 时会读取，不用记 NodePoolId）")

    if args.wait:
        _wait_ready(cli, node_pool_id, timeout=args.wait_timeout)
    return 0


def _wait_ready(cli, node_pool_id: str, timeout: float) -> None:
    from tencentcloud.tke.v20180525 import models

    print("等待节点池状态变为 Normal 且有节点加入...")
    started = time.time()
    while time.time() - started < timeout:
        req = models.DescribeClusterNodePoolDetailRequest()
        req.ClusterId = CLUSTER_ID
        req.NodePoolId = node_pool_id
        rsp = cli.DescribeClusterNodePoolDetail(req)
        np = rsp.NodePool
        status = np.LifeState
        print(f"  [{int(time.time()-started)}s] LifeState={status}")
        if status == "normal":
            print("节点池已就绪（Normal）。用 `kubectl get nodes` 确认节点真正 Ready。")
            return
        time.sleep(15)
    print(f"⚠️ 等待超时（{timeout}s），请手动用 `status` 子命令或控制台检查")


def cmd_status(args: argparse.Namespace) -> int:
    from tencentcloud.tke.v20180525 import models

    if not STATE_FILE.exists():
        print("没有本地状态记录，直接列出集群所有节点池：")
        cli = _tke_client()
        req = models.DescribeClusterNodePoolsRequest()
        req.ClusterId = CLUSTER_ID
        rsp = cli.DescribeClusterNodePools(req)
        for np in rsp.NodePoolSet or []:
            print(f"  {np.NodePoolId}  {np.Name}  LifeState={np.LifeState}")
        return 0

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    cli = _tke_client()
    req = models.DescribeClusterNodePoolDetailRequest()
    req.ClusterId = state["cluster_id"]
    req.NodePoolId = state["node_pool_id"]
    rsp = cli.DescribeClusterNodePoolDetail(req)
    np = rsp.NodePool
    print(f"NodePoolId={np.NodePoolId}  Name={np.Name}  LifeState={np.LifeState}")
    print(f"DesiredCapacity={np.AutoscalingGroupStatus} NodeCountSummary={np.NodeCountSummary}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    from tencentcloud.tke.v20180525 import models

    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        node_pool_id = state["node_pool_id"]
        cluster_id = state["cluster_id"]
    elif args.node_pool_id:
        node_pool_id = args.node_pool_id
        cluster_id = CLUSTER_ID
    else:
        print("找不到 .gpu_nodepool_state.json，也没传 --node-pool-id，无法删除")
        return 1

    cli = _tke_client()
    req = models.DeleteClusterNodePoolRequest()
    req.ClusterId = cluster_id
    req.NodePoolIds = [node_pool_id]
    req.KeepInstance = False  # 连底层 CVM 一起销毁，彻底停止计费
    cli.DeleteClusterNodePool(req)
    print(f"已提交删除请求：NodePoolId={node_pool_id}（含底层 CVM 一起销毁）")

    if STATE_FILE.exists():
        STATE_FILE.unlink()
    return 0


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description="TKE GPU 节点池管理（Phase 3 专用，谨慎操作）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("--node-password", default="SweRl2026!Gpu")
    p_create.add_argument("--wait", action="store_true", default=True)
    p_create.add_argument("--no-wait", dest="wait", action="store_false")
    p_create.add_argument("--wait-timeout", type=float, default=600)
    p_create.set_defaults(func=cmd_create)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--node-pool-id", default=None)
    p_delete.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
