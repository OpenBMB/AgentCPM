#!/usr/bin/env python3
"""
修复 MongoDB 副本集配置，将内部地址改为外部可访问地址
"""
import pymongo
import sys

# 配置
MONGO_HOST = "11.11.23.3"
MONGO_PORT = 27018
EXTERNAL_HOST = "11.11.23.3"  # 外部可访问的地址
EXTERNAL_PORT = 27018
REPLICA_SET_NAME = "rs0"
USERNAME = "root"
PASSWORD = "password"

def fix_replica_set():
    """修复副本集配置"""
    connection_string = f"mongodb://{USERNAME}:{PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/admin?directConnection=true"
    
    print(f"连接到 MongoDB: {MONGO_HOST}:{MONGO_PORT}")
    print("-" * 60)
    
    try:
        client = pymongo.MongoClient(
            connection_string,
            serverSelectionTimeoutMS=10000
        )
        
        # 检查当前副本集状态
        try:
            rs_status = client.admin.command('replSetGetStatus')
            print(f"✓ 当前副本集: {rs_status.get('set', 'N/A')}")
            print(f"✓ 成员数量: {len(rs_status.get('members', []))}")
            print("\n当前成员配置:")
            for member in rs_status.get('members', []):
                print(f"  - {member.get('name', 'N/A')} (状态: {member.get('stateStr', 'N/A')})")
        except Exception as e:
            print(f"⚠️  无法获取副本集状态: {e}")
            return
        
        # 获取配置
        config = client.admin.command('replSetGetConfig')
        current_config = config['config']
        
        # 检查是否需要更新
        needs_update = False
        for member in current_config['members']:
            if '127.0.0.1' in member['host'] or member['host'].startswith('mongo-'):
                needs_update = True
                break
        
        if not needs_update:
            print("\n✓ 副本集配置已正确，无需更新")
            return
        
        # 更新配置
        print("\n开始更新副本集配置...")
        new_members = []
        for i, member in enumerate(current_config['members']):
            new_member = member.copy()
            # 将内部地址替换为外部地址
            if i == 0:  # 主节点
                new_member['host'] = f"{EXTERNAL_HOST}:{EXTERNAL_PORT}"
            else:
                # 如果有其他节点，也需要更新
                new_member['host'] = f"{EXTERNAL_HOST}:{EXTERNAL_PORT}"
            new_members.append(new_member)
            print(f"  更新成员 {i}: {member['host']} -> {new_member['host']}")
        
        current_config['members'] = new_members
        current_config['version'] += 1
        
        # 应用新配置
        try:
            client.admin.command({
                'replSetReconfig': current_config
            })
            print("\n✓ 副本集配置更新成功！")
            print("  注意: 如果这是单节点副本集，可能需要重新初始化")
        except Exception as e:
            print(f"\n✗ 更新配置失败: {e}")
            print("  可能需要手动执行以下命令:")
            print(f"  rs.reconfig({current_config})")
        
        client.close()
        
    except Exception as e:
        print(f"✗ 操作失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    fix_replica_set()


