import pulumi
import pulumi_alicloud as alicloud
from typing import Optional, Dict, Any
import config


def create_vpc() -> alicloud.vpc.Network:
    """
    Create a VPC for the Freqtrade infrastructure.

    Returns:
        alicloud.vpc.Network: The created VPC resource
    """
    return alicloud.vpc.Network(
        config.config.vpc_name,
        cidr_block=config.config.vpc_cidr,
        description="VPC for freqtrade bot",
        vpc_name=config.config.vpc_name,
        enable_ipv6=False,
        tags={
            "Name": config.config.vpc_name,
            "Environment": pulumi.get_stack(),
            "ManagedBy": "Pulumi",
            "Project": "freqtrade"
        },
    )


def create_vswitch(vpc_id: pulumi.Input[str], zone_id: str) -> alicloud.vpc.Switch:
    """
    Create a VSwitch in the specified VPC and zone.

    Args:
        vpc_id: The ID of the VPC where the VSwitch will be created
        zone_id: The zone ID where the VSwitch will be created

    Returns:
        alicloud.vpc.Switch: The created VSwitch resource
    """
    return alicloud.vpc.Switch(
        f"{config.config.vpc_name}-vsw",
        vpc_id=vpc_id,
        cidr_block=config.config.vswitch_cidr,
        vswitch_name=f"{config.config.vpc_name}-vsw",
        zone_id=zone_id,
        tags={
            "Name": f"{config.config.vpc_name}-vsw",
            "Environment": pulumi.get_stack(),
            "ManagedBy": "Pulumi",
            "Project": "freqtrade"
        },
    )


def get_vpc_outputs(vpc: alicloud.vpc.Network, vswitch: Optional[alicloud.vpc.Switch] = None) -> Dict[str, Any]:
    """
    Get the VPC outputs for export.

    Args:
        vpc: The VPC resource

    Returns:
        Dict[str, pulumi.Output[str]]: Dictionary of VPC outputs
    """
    outputs = {
        "vpc_id": vpc.id,
        "vpc_name": vpc.vpc_name,
        "cidr_block": vpc.cidr_block,
    }

    if vswitch:
        outputs.update({
            "vswitch_id": vswitch.id,
            "vswitch_name": vswitch.vswitch_name,
            "vswitch_cidr": vswitch.cidr_block,
            "zone_id": vswitch.zone_id
        })

    return outputs


# Export VPC resources if this module is run directly

