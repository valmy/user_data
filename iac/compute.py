from typing import Dict, List, Optional, Tuple, Union
import pulumi
import pulumi_alicloud as alicloud
import config


def create_eip(instance_name: str) -> alicloud.ecs.EipAddress:
    """
    Create an Elastic IP address.
    
    Args:
        instance_name: Name of the instance this EIP will be associated with
        
    Returns:
        alicloud.ecs.EipAddress: The created EIP resource
    """
    return alicloud.ecs.EipAddress(
        f"{instance_name}-eip",
        address_name=f"{instance_name}-eip",
        isp="BGP",
        internet_charge_type="PayByTraffic",
        payment_type="PayAsYouGo",
        tags={
            "Name": f"{instance_name}-eip",
            "Environment": pulumi.get_stack(),
            "ManagedBy": "Pulumi",
            "Project": "freqtrade"
        },
    )


def create_ecs_instance(
    security_group_ids: List[pulumi.Input[str]],
    vswitch_id: Optional[pulumi.Input[str]] = None,
    allocate_public_ip: bool = True,
    use_eip: bool = False,
) -> Tuple[alicloud.ecs.Instance, Optional[alicloud.ecs.EipAddress]]:
    """
    Create an ECS instance for Freqtrade.
    
    Args:
        security_group_ids: List of security group IDs to attach to the instance
        vswitch_id: Optional VSwitch ID. If not provided, will use default VSwitch.
        allocate_public_ip: Whether to assign a public IP to the instance
        use_eip: Whether to create and associate an Elastic IP
        
    Returns:
        Tuple containing:
            - alicloud.ecs.Instance: The created ECS instance resource
            - Optional[alicloud.ecs.EipAddress]: The EIP if use_eip is True, else None
    """
    # Create EIP first if requested
    eip = None
    if use_eip:
        eip = create_eip(config.config.ecs_instance_name)
        # If using EIP, we don't need to allocate a public IP
        allocate_public_ip = False

    instance_args = {
        "instance_name": config.config.ecs_instance_name,
        "instance_type": config.config.instance_type,
        "image_id": config.config.image_id,
        "security_groups": security_group_ids,
        "key_name": config.config.key_name,
        "internet_max_bandwidth_out": config.config.internet_max_bandwidth_out if allocate_public_ip else 0,
        "instance_charge_type": "PostPaid",
        "system_disk_category": config.config.system_disk_category,
        "system_disk_size": config.config.system_disk_size,  
        "tags": {
            "Name": config.config.ecs_instance_name,
            "Environment": pulumi.get_stack(),
            "ManagedBy": "Pulumi",
            "Project": "freqtrade"
        },
    }

    # Add VSwitch if provided
    if not vswitch_id and hasattr(config.config, 'vswitch_id'):
        vswitch_id = config.config.vswitch_id
        
    if vswitch_id:
        instance_args["vswitch_id"] = vswitch_id
    
    instance = alicloud.ecs.Instance(
        config.config.ecs_instance_name,
        **instance_args
    )
    
    # Associate EIP if created
    if eip is not None:
        alicloud.ecs.EipAssociation(
            f"{config.config.ecs_instance_name}-eip-assoc",
            allocation_id=eip.id,
            instance_id=instance.id,
        )
    
    return instance, eip


def get_compute_outputs(instance: alicloud.ecs.Instance, eip: Optional[alicloud.ecs.EipAddress] = None) -> Dict[str, pulumi.Output[str]]:
    """
    Get the compute outputs for the given instance and optional EIP.
    
    Args:
        instance: The ECS instance
        eip: Optional EIP associated with the instance
        
    Returns:
        Dict of output names to output values
    """
    outputs = {
        "instance_id": instance.id,
        "instance_public_ip": instance.public_ip,
        "instance_private_ip": instance.private_ip,
        "ssh_command": pulumi.Output.format(
            'ssh -i {key_file} {username}@{ip}',
            key_file=config.config.ssh_private_key_path,
            username=config.config.ssh_username,
            ip=instance.public_ip if instance.public_ip else instance.private_ip
        )
    }
    
    if eip is not None:
        outputs["eip_address"] = eip.ip_address
        outputs["ssh_command_eip"] = pulumi.Output.format(
            'ssh -i {key_file} {username}@{ip}',
            key_file=config.config.ssh_private_key_path,
            username=config.config.ssh_username,
            ip=eip.ip_address
        )
    
    return outputs


# Export compute resources if this module is run directly
if __name__ == "__main__":
    # This is just for testing - in practice, security group IDs should be passed as parameters
    instance = create_ecs_instance(security_group_ids=["sg-test-id"])
    outputs = get_compute_outputs(instance)
    for key, value in outputs.items():
        pulumi.export(f"compute_{key}", value)
