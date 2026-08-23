#!/usr/bin/env python3
"""
云资源探测（只读）—— 题目四专用
================================

专门回答 Phase 0 要确认的三件事：
    · TKE：账号里有没有现成集群？有没有 GPU 节点池？GPU 机型在当前地域有没有货？
    · COS：有没有现成 bucket 可以用来传 tracing？
    · CFS：有没有现成文件系统？（如果 SandBox 和 TKE 需要共享盘）

凭证只读取本项目目录下的 `.env`（题目四独立配置，不再回退读取其他课题目录）。
云账号 / TKE 集群 / COS bucket 等基础设施资源可以与账号内其他项目共用，
但凭证文件本身在本项目内自成一份，不依赖外部路径是否存在。

安全约定：
    · 凭证只从环境变量 / .env 读取，绝不写进代码
    · SecretId 只回显前 8 位，SecretKey 完全不回显
    · 全程只调 Describe/Get/List 类只读接口，不创建、不修改、不删除任何资源

用法：
    python scripts/probe_cloud.py                # 全部探测（cam/ags/tke/cfs/cos）
    python scripts/probe_cloud.py --only tke      # 只探某一项
    python scripts/probe_cloud.py --json out.json # 顺带写结构化结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 题目四：强化学习/

MY_PREFIX = "swe-synth"  # 沿用统一命名前缀，方便从一堆资源里认出「自己的」


# ---------------------------------------------------------------- 基础设施

def load_env() -> str:
    """只读本项目目录下的 .env（题目四独立配置）。返回实际使用的路径。"""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return ""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except ImportError:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    return path


def env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


def title(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def kv(k: str, v) -> None:
    print(f"  {k:<26} {v}")


def explain_error(e: Exception) -> str:
    msg = str(e)
    code = getattr(e, "code", "") or ""
    if "AuthFailure.SignatureFailure" in msg or ("SecretId" in msg and "not exist" in msg):
        return "密钥无效或已禁用 → 去 CAM → 访问密钥 确认 SecretId/SecretKey"
    if "UnauthorizedOperation" in code or "UnauthorizedOperation" in msg:
        return "子用户无此接口权限 → 需要在自研上云平台补提权限单"
    if "AuthFailure" in code or "AuthFailure" in msg:
        return "鉴权失败 → 检查密钥、系统时间是否正确"
    if "ResourceNotFound" in msg or "InvalidParameter" in msg:
        return "资源不存在或参数不对（也可能是该产品尚未开通）"
    if "not open" in msg.lower() or "未开通" in msg:
        return "该云产品尚未开通 → 去控制台开通，或提单申请"
    return "未归类错误，把完整信息发给导师定位更快"


def client_of(module_name: str, client_cls: str, version: str, region: str):
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile

    sid, skey = env("TENCENTCLOUD_SECRET_ID"), env("TENCENTCLOUD_SECRET_KEY")
    cred = credential.Credential(sid, skey)
    hp = HttpProfile(reqTimeout=30)
    cp = ClientProfile(httpProfile=hp)

    mod = __import__(f"tencentcloud.{module_name}.{version}.{module_name}_client",
                     fromlist=[client_cls])
    cls = getattr(mod, client_cls)
    return cls(cred, region, cp)


def models_of(module_name: str, version: str):
    return __import__(f"tencentcloud.{module_name}.{version}.models", fromlist=["models"])


# ---------------------------------------------------------------- 1. 身份（简版）

def probe_cam(region: str) -> dict:
    title("1. 身份与权限（CAM）")
    out: dict = {}
    try:
        cli = client_of("cam", "CamClient", "v20190116", region)
        m = models_of("cam", "v20190116")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 初始化失败：{e}")
        return out

    uin = owner_uin = None
    try:
        rsp = cli.GetUserAppId(m.GetUserAppIdRequest())
        uin, owner_uin = rsp.Uin, rsp.OwnerUin
        out["uin"], out["owner_uin"], out["app_id"] = uin, owner_uin, rsp.AppId
        kv("子用户 Uin", uin)
        kv("主账号 OwnerUin", owner_uin)
        kv("AppId", rsp.AppId)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ GetUserAppId 失败：{e}")
        print(f"     → {explain_error(e)}")
        return out

    try:
        req = m.ListAttachedUserAllPoliciesRequest()
        req.TargetUin = int(uin)
        req.AttachType = 0
        req.Page, req.Rp = 1, 200
        rsp = cli.ListAttachedUserAllPolicies(req)
        names = [p.PolicyName for p in (rsp.PolicyList or [])]
        out["policies"] = names
        kv("已关联策略数", len(names))

        need = {
            "TKE 集群管理": ("tke", "ckafka" and "tke", "cvm"),
            "COS 对象存储": ("cos",),
            "CFS 文件存储": ("cfs",),
        }
        print("\n  能力核验（关键字匹配，仅作参考）：")
        joined = " ".join(names).lower()
        full = "AdministratorAccess" in joined or "QCloudResourceFullAccess" in joined
        for cap, kws in need.items():
            hit = any(k in joined for k in kws)
            mark = "✅" if (hit or full) else "❓"
            kv(f"    {mark} {cap}", "疑似已授权" if (hit or full) else "未在策略名中识别到（可能靠用户组继承）")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  策略列表查询失败：{e}")
        print(f"     → {explain_error(e)}（不影响后续探测，控制台也能看）")

    return out


# ---------------------------------------------------------------- 2. AGS 配额（简版，仅看余量）

def probe_ags_quota(region: str) -> dict:
    title("2. AGS 沙箱工具配额（复用课题三资产的前提）")
    out: dict = {}
    try:
        cli = client_of("ags", "AgsClient", "v20250920", region)
        m = models_of("ags", "v20250920")
        req = m.DescribeSandboxToolListRequest()
        req.Offset, req.Limit = 0, 100
        rsp = cli.DescribeSandboxToolList(req)
        tools = rsp.SandboxToolSet or []
        total = rsp.TotalCount
        out["tools_total"] = total
        out["tools"] = [getattr(t, "ToolName", "?") for t in tools]
        kv("当前已注册沙箱工具数", total)
        for t in tools:
            print(f"      · {getattr(t, 'ToolName', '?')}  状态={getattr(t, 'Status', '?')}")
        quota = 10
        remain = quota - total
        kv("配额上限（经验值）", quota)
        kv("剩余可注册数", remain)
        if remain <= 2:
            print("     ⚠️  剩余配额紧张 → Phase 2 需「用前建、验证完删」滚动注册，不能一次性全量注册")
        else:
            print("     ✅ 配额充裕")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 查询失败：{e}")
        print(f"     → {explain_error(e)}")
    return out


# ---------------------------------------------------------------- 3. TKE

def probe_tke(region: str) -> dict:
    title("3. 容器服务 TKE（训练侧 GPU 集群现状）")
    out: dict = {"clusters": []}
    try:
        cli = client_of("tke", "TkeClient", "v20180525", region)
        m = models_of("tke", "v20180525")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 初始化失败：{e}")
        return out

    clusters = []
    try:
        req = m.DescribeClustersRequest()
        req.Limit, req.Offset = 100, 0
        rsp = cli.DescribeClusters(req)
        clusters = rsp.Clusters or []
        kv("现有集群数（当前地域）", rsp.TotalCount)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ DescribeClusters 失败：{e}")
        print(f"     → {explain_error(e)}")
        return out

    if not clusters:
        print("\n  → 当前地域没有现成 TKE 集群，需要新建（控制台「容器服务」→ 新建集群，")
        print("    建议：标准集群 + 1 个 GPU 节点池，按量计费，训练完立即缩容到 0 或删除节点池）")
        print("    也可以先看看其它地域是否有集群（--region 换一个试试）")
        return out

    for c in clusters:
        cid = c.ClusterId
        cname = getattr(c, "ClusterName", "?")
        status = getattr(c, "ClusterStatus", "?")
        version = getattr(c, "ClusterVersion", "?")
        print(f"\n      集群：{cname}  ({cid})")
        kv("        状态", status)
        kv("        K8s 版本", version)
        info = {"id": cid, "name": cname, "status": status, "version": version, "nodes": []}

        # 3.1 节点列表 + 是否含 GPU（交叉查 CVM 机型）
        try:
            nreq = m.DescribeClusterInstancesRequest()
            nreq.ClusterId = cid
            nreq.Limit, nreq.Offset = 100, 0
            nrsp = cli.DescribeClusterInstances(nreq)
            insts = nrsp.InstanceSet or []
            kv("        节点数", nrsp.TotalCount)
            info["node_count"] = nrsp.TotalCount
            ids = [i.InstanceId for i in insts if getattr(i, "InstanceId", None)]
            info["nodes"] = ids
            if ids:
                gpu_nodes = _cross_check_gpu(region, ids)
                if gpu_nodes:
                    print(f"        ✅ 发现 {len(gpu_nodes)} 个 GPU 节点：")
                    for n in gpu_nodes:
                        print(f"           · {n['instance_id']}  {n['instance_type']}")
                    info["gpu_nodes"] = gpu_nodes
                else:
                    print("        ❓ 未发现 GPU 节点（都是 CPU 机型）→ 需要新增 GPU 节点池")
            else:
                print("        （无节点，是个空集群）")
        except Exception as e:  # noqa: BLE001
            print(f"        ⚠️  节点查询失败：{e}")

        # 3.2 节点池（如果走节点池管理，GPU 伸缩更方便）
        try:
            preq = m.DescribeClusterNodePoolsRequest()
            preq.ClusterId = cid
            prsp = cli.DescribeClusterNodePools(preq)
            pools = prsp.NodePoolSet or []
            kv("        节点池数", len(pools))
            info["node_pools"] = [getattr(p, "Name", "?") for p in pools]
            for p in pools:
                print(f"           节点池：{getattr(p, 'Name', '?')}  "
                      f"期望节点数={getattr(p, 'DesiredNodesNum', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"        ⚠️  节点池查询失败（可能该地域/版本不支持或无权限）：{e}")

        out["clusters"].append(info)

    return out


def _cross_check_gpu(region: str, instance_ids: list) -> list:
    """用 CVM DescribeInstances 交叉查节点机型，判断是否为 GPU 机型。"""
    try:
        cli = client_of("cvm", "CvmClient", "v20170312", region)
        m = models_of("cvm", "v20170312")
        req = m.DescribeInstancesRequest()
        req.InstanceIds = instance_ids[:100]
        rsp = cli.DescribeInstances(req)
        result = []
        for i in rsp.InstanceSet or []:
            itype = getattr(i, "InstanceType", "") or ""
            # 腾讯云 GPU 机型系列常见前缀：GN / GT / PNV
            if any(itype.upper().startswith(p) for p in ("GN", "GT", "PNV")):
                result.append({"instance_id": i.InstanceId, "instance_type": itype})
        return result
    except Exception:  # noqa: BLE001
        return []


def probe_gpu_stock(region: str) -> dict:
    """当前地域 GPU 机型是否有货 —— 决定新建节点池能不能立刻拿到卡，避免卡在排队。"""
    title("3b. 当前地域 GPU 机型库存（供新建 GPU 节点池参考）")
    out: dict = {}
    try:
        cli = client_of("cvm", "CvmClient", "v20170312", region)
        m = models_of("cvm", "v20170312")
        req = m.DescribeZoneInstanceConfigInfosRequest()
        rsp = cli.DescribeZoneInstanceConfigInfos(req)
        items = rsp.InstanceTypeQuotaSet or []
        gpu_items = [x for x in items if (getattr(x, "Gpu", 0) or 0) > 0]
        sellable = [x for x in gpu_items if getattr(x, "Status", "") == "SELL"]
        kv("GPU 机型规格总数", len(gpu_items))
        kv("其中可售（SELL）", len(sellable))
        seen = set()
        print("\n  可售 GPU 机型样例（去重，最多显示 10 个）：")
        shown = 0
        for x in sellable:
            key = (x.InstanceType, x.Zone)
            if x.InstanceType in seen:
                continue
            seen.add(x.InstanceType)
            print(f"      · {x.InstanceType:<20} 可用区={x.Zone:<16} "
                  f"GPU={x.Gpu} CPU={x.Cpu} 内存={x.Memory}GB")
            shown += 1
            if shown >= 10:
                break
        out["gpu_sellable_types"] = sorted(seen)
        if not sellable:
            print("     ❌ 当前地域没有可售 GPU 机型 → 换地域，或提单申请库存")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  查询失败：{e}")
        print(f"     → {explain_error(e)}")
    return out


# ---------------------------------------------------------------- 4. CFS

def probe_cfs(region: str) -> dict:
    title("4. 文件存储 CFS（SandBox↔TKE 共享盘现状）")
    out: dict = {"filesystems": []}
    try:
        cli = client_of("cfs", "CfsClient", "v20190719", region)
        m = models_of("cfs", "v20190719")
        req = m.DescribeCfsFileSystemsRequest()
        rsp = cli.DescribeCfsFileSystems(req)
        fss = rsp.FileSystems or []
        kv("现有文件系统数（当前地域）", len(fss))
        for f in fss:
            fsid = getattr(f, "FileSystemId", "?")
            name = getattr(f, "FsName", "?")
            status = getattr(f, "LifeCycleState", "?")
            proto = getattr(f, "Protocol", "?")
            size = getattr(f, "SizeByte", None)
            print(f"\n      文件系统：{name}  ({fsid})")
            kv("        状态", status)
            kv("        协议", proto)
            if size is not None:
                kv("        已用容量", f"{size / 1024**3:.2f} GB")
            out["filesystems"].append({"id": fsid, "name": name, "status": status})
        if not fss:
            print("\n  → 当前地域没有现成 CFS，若方案不依赖共享盘（用 COS 传 tracing 即可），")
            print("    可以不用新建 CFS，跳过这一项，简化链路")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 查询失败：{e}")
        print(f"     → {explain_error(e)}")
    return out


# ---------------------------------------------------------------- 5. COS

def probe_cos(region: str) -> dict:
    title("5. 对象存储 COS（SandBox → TKE 传 tracing 的通道）")
    out: dict = {"buckets": []}
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError:
        print("  ❌ 缺少 cos-python-sdk-v5：pip install cos-python-sdk-v5")
        return out

    sid, skey = env("TENCENTCLOUD_SECRET_ID"), env("TENCENTCLOUD_SECRET_KEY")
    try:
        cfg = CosConfig(Region=region, SecretId=sid, SecretKey=skey)
        cli = CosS3Client(cfg)
        rsp = cli.list_buckets()
        buckets = (rsp.get("Buckets") or {}).get("Bucket") or []
        kv("账号内 bucket 总数（全地域）", len(buckets))
        mine = []
        for b in buckets:
            name = b.get("Name", "?")
            loc = b.get("Location", "?")
            print(f"      · {name}  地域={loc}  创建于={b.get('CreationDate', '?')}")
            if MY_PREFIX in name.lower():
                mine.append(name)
        out["buckets"] = [b.get("Name") for b in buckets]
        if mine:
            print(f"\n     ✅ 已有本课题相关 bucket：{mine}")
        else:
            print(f"\n     → 没有名字含 {MY_PREFIX} 的 bucket，建议新建一个专用 bucket")
            print(f"       （控制台一键创建，几秒钟；或 CreateBucket API，Bucket 名建议：")
            print(f"        {MY_PREFIX}-rl-<你的APPID>-{region}）")
        out["has_mine"] = bool(mine)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 查询失败：{e}")
        print(f"     → {explain_error(e)}")
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="云资源探测（只读，不创建任何资源）· 题目四专用")
    ap.add_argument("--only", choices=["cam", "ags", "tke", "gpu", "cfs", "cos"], help="只探测某一项")
    ap.add_argument("--region", default=None, help="覆盖地域，默认读 .env 的 TENCENTCLOUD_REGION")
    ap.add_argument("--json", metavar="PATH", help="把结构化结果写入 JSON 文件")
    args = ap.parse_args()

    env_path = load_env()
    sid, skey = env("TENCENTCLOUD_SECRET_ID"), env("TENCENTCLOUD_SECRET_KEY")
    region = args.region or env("TENCENTCLOUD_REGION", "ap-guangzhou")

    if not sid or not skey:
        print("❌ 未配置 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY")
        print(f"   已尝试读取：{env_path or '(未找到 .env)'}")
        print("   请在题目四目录下 cp .env.example .env 并填入真实值")
        return 1

    print(f"使用密钥 {sid[:8]}****（SecretKey 不回显）")
    print(f".env 来源：{env_path}")
    print(f"地域：{region}")

    try:
        import tencentcloud  # noqa: F401
    except ImportError:
        print("❌ 缺少 tencentcloud-sdk-python：pip install tencentcloud-sdk-python")
        return 1

    probes = {
        "cam": lambda r: probe_cam(r),
        "ags": lambda r: probe_ags_quota(r),
        "tke": lambda r: probe_tke(r),
        "gpu": lambda r: probe_gpu_stock(r),
        "cfs": lambda r: probe_cfs(r),
        "cos": lambda r: probe_cos(r),
    }
    names = [args.only] if args.only else list(probes)

    result: dict = {"region": region}
    for n in names:
        try:
            result[n] = probes[n](region)
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ {n} 探测异常：{type(e).__name__}: {e}")
            result[n] = {"error": str(e)}

    title("探测完毕 · 结论会写进 PROGRESS.md")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  结构化结果已写入 {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
