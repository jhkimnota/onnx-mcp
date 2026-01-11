"""ONNX MCP Server - Tools for inspecting ONNX models."""

import json
import os
from collections import Counter
from typing import Any

import numpy as np
import onnx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("onnx-mcp-server")

# ONNX data type mapping
ONNX_DTYPE_MAP = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
}

DTYPE_BYTES = {
    "FLOAT": 4,
    "UINT8": 1,
    "INT8": 1,
    "UINT16": 2,
    "INT16": 2,
    "INT32": 4,
    "INT64": 8,
    "BOOL": 1,
    "FLOAT16": 2,
    "DOUBLE": 8,
    "UINT32": 4,
    "UINT64": 8,
    "BFLOAT16": 2,
}


def load_model(file_path: str) -> onnx.ModelProto:
    """Load an ONNX model from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ONNX file not found: {file_path}")
    return onnx.load(file_path)


def get_tensor_shape(tensor_type) -> list[str | int]:
    """Extract shape from tensor type."""
    shape = []
    if tensor_type.HasField("tensor_type"):
        for dim in tensor_type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")
    return shape


def get_tensor_dtype(tensor_type) -> str:
    """Extract data type from tensor type."""
    if tensor_type.HasField("tensor_type"):
        return ONNX_DTYPE_MAP.get(tensor_type.tensor_type.elem_type, "UNKNOWN")
    return "UNKNOWN"


def format_shape(shape: list) -> str:
    """Format shape as string."""
    return f"[{', '.join(str(d) for d in shape)}]"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="get_model_info",
            description="Get basic metadata of an ONNX model (name, IR version, opset, producer, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_inputs_outputs",
            description="Get input and output tensor information (name, type, shape)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_graph_structure",
            description="Get the graph structure with nodes and their connections",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_node_details",
            description="Get detailed information about a specific node",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    },
                    "node_name": {
                        "type": "string",
                        "description": "Name of the node to inspect",
                    },
                    "node_index": {
                        "type": "integer",
                        "description": "Index of the node (0-based). Used if node_name is not provided.",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="visualize_graph",
            description="Generate a Mermaid diagram of the graph structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum number of nodes to include (default: 50)",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_weight_statistics",
            description="Get weight/parameter statistics (count, shapes, value statistics)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="estimate_model_complexity",
            description="Estimate memory usage and FLOPs of the model",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the ONNX file",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for estimation (default: 1)",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "get_model_info":
            result = get_model_info(arguments["file_path"])
        elif name == "get_inputs_outputs":
            result = get_inputs_outputs(arguments["file_path"])
        elif name == "get_graph_structure":
            result = get_graph_structure(arguments["file_path"])
        elif name == "get_node_details":
            result = get_node_details(
                arguments["file_path"],
                arguments.get("node_name"),
                arguments.get("node_index"),
            )
        elif name == "visualize_graph":
            result = visualize_graph(
                arguments["file_path"], arguments.get("max_nodes", 50)
            )
        elif name == "get_weight_statistics":
            result = get_weight_statistics(arguments["file_path"])
        elif name == "estimate_model_complexity":
            result = estimate_model_complexity(
                arguments["file_path"], arguments.get("batch_size", 1)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def get_model_info(file_path: str) -> dict:
    """Get basic model metadata."""
    model = load_model(file_path)

    opset_imports = []
    for opset in model.opset_import:
        opset_imports.append(
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
        )

    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value

    return {
        "model_name": model.graph.name,
        "description": model.doc_string or None,
        "ir_version": model.ir_version,
        "opset_imports": opset_imports,
        "producer_name": model.producer_name or None,
        "producer_version": model.producer_version or None,
        "domain": model.domain or None,
        "model_version": model.model_version,
        "metadata": metadata if metadata else None,
    }


def get_inputs_outputs(file_path: str) -> dict:
    """Get input and output tensor information."""
    model = load_model(file_path)
    graph = model.graph

    inputs = []
    for inp in graph.input:
        inputs.append(
            {
                "name": inp.name,
                "dtype": get_tensor_dtype(inp.type),
                "shape": get_tensor_shape(inp.type),
            }
        )

    outputs = []
    for out in graph.output:
        outputs.append(
            {
                "name": out.name,
                "dtype": get_tensor_dtype(out.type),
                "shape": get_tensor_shape(out.type),
            }
        )

    return {"inputs": inputs, "outputs": outputs}


def get_graph_structure(file_path: str) -> dict:
    """Get graph structure with nodes and connections."""
    model = load_model(file_path)
    graph = model.graph

    nodes = []
    op_type_counts = Counter()

    for i, node in enumerate(graph.node):
        op_type_counts[node.op_type] += 1
        nodes.append(
            {
                "index": i,
                "name": node.name or f"node_{i}",
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )

    return {
        "total_nodes": len(nodes),
        "op_type_statistics": dict(op_type_counts.most_common()),
        "nodes": nodes,
    }


def get_node_details(
    file_path: str, node_name: str | None = None, node_index: int | None = None
) -> dict:
    """Get detailed information about a specific node."""
    model = load_model(file_path)
    graph = model.graph

    node = None
    if node_name:
        for n in graph.node:
            if n.name == node_name:
                node = n
                break
        if node is None:
            return {"error": f"Node with name '{node_name}' not found"}
    elif node_index is not None:
        if 0 <= node_index < len(graph.node):
            node = graph.node[node_index]
        else:
            return {"error": f"Node index {node_index} out of range (0-{len(graph.node) - 1})"}
    else:
        return {"error": "Either node_name or node_index must be provided"}

    attributes = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attributes[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attributes[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attributes[attr.name] = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.FLOATS:
            attributes[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            attributes[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            attributes[attr.name] = [s.decode("utf-8") for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSOR:
            attributes[attr.name] = f"<tensor: {attr.t.dims}>"
        elif attr.type == onnx.AttributeProto.GRAPH:
            attributes[attr.name] = f"<subgraph: {attr.g.name}>"
        else:
            attributes[attr.name] = f"<type: {attr.type}>"

    return {
        "name": node.name,
        "op_type": node.op_type,
        "domain": node.domain or "ai.onnx",
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": attributes,
    }


def visualize_graph(file_path: str, max_nodes: int = 50) -> dict:
    """Generate a Mermaid diagram of the graph structure."""
    model = load_model(file_path)
    graph = model.graph

    lines = ["graph TD"]
    node_count = len(graph.node)

    if node_count > max_nodes:
        lines.append(f"    %% Showing first {max_nodes} of {node_count} nodes")

    # Track tensor producers
    tensor_producers: dict[str, str] = {}
    for inp in graph.input:
        tensor_producers[inp.name] = f"input_{inp.name}"
        safe_name = inp.name.replace("-", "_").replace(".", "_")
        lines.append(f"    input_{safe_name}[/\"{inp.name}\"/]")

    # Add nodes
    for i, node in enumerate(graph.node[:max_nodes]):
        node_id = node.name or f"node_{i}"
        safe_id = node_id.replace("-", "_").replace(".", "_")
        lines.append(f"    {safe_id}[\"{node.op_type}\"]")

        for output in node.output:
            tensor_producers[output] = safe_id

    # Add edges
    for i, node in enumerate(graph.node[:max_nodes]):
        node_id = node.name or f"node_{i}"
        safe_id = node_id.replace("-", "_").replace(".", "_")

        for inp in node.input:
            if inp in tensor_producers:
                source = tensor_producers[inp]
                lines.append(f"    {source} --> {safe_id}")

    # Add outputs
    for out in graph.output:
        safe_name = out.name.replace("-", "_").replace(".", "_")
        lines.append(f"    output_{safe_name}[\\\" {out.name}\"/]")
        if out.name in tensor_producers:
            source = tensor_producers[out.name]
            lines.append(f"    {source} --> output_{safe_name}")

    mermaid_code = "\n".join(lines)

    return {
        "format": "mermaid",
        "total_nodes": node_count,
        "displayed_nodes": min(max_nodes, node_count),
        "diagram": mermaid_code,
    }


def get_weight_statistics(file_path: str) -> dict:
    """Get weight/parameter statistics."""
    model = load_model(file_path)
    graph = model.graph

    initializers = []
    total_params = 0
    total_bytes = 0

    for init in graph.initializer:
        tensor = onnx.numpy_helper.to_array(init)
        param_count = int(np.prod(tensor.shape))
        dtype_name = ONNX_DTYPE_MAP.get(init.data_type, "UNKNOWN")
        bytes_per_elem = DTYPE_BYTES.get(dtype_name, 4)
        memory_bytes = param_count * bytes_per_elem

        total_params += param_count
        total_bytes += memory_bytes

        stats = {
            "name": init.name,
            "shape": list(tensor.shape),
            "dtype": dtype_name,
            "param_count": param_count,
            "memory_bytes": memory_bytes,
        }

        if tensor.size > 0 and np.issubdtype(tensor.dtype, np.number):
            stats["value_stats"] = {
                "min": float(np.min(tensor)),
                "max": float(np.max(tensor)),
                "mean": float(np.mean(tensor)),
                "std": float(np.std(tensor)),
            }

        initializers.append(stats)

    return {
        "total_parameters": total_params,
        "total_memory_bytes": total_bytes,
        "total_memory_mb": round(total_bytes / (1024 * 1024), 2),
        "initializer_count": len(initializers),
        "initializers": initializers,
    }


def estimate_model_complexity(file_path: str, batch_size: int = 1) -> dict:
    """Estimate memory usage and FLOPs."""
    model = load_model(file_path)
    graph = model.graph

    # Get file size
    file_size = os.path.getsize(file_path)

    # Get weight memory from initializers
    weight_memory = 0
    for init in graph.initializer:
        tensor = onnx.numpy_helper.to_array(init)
        dtype_name = ONNX_DTYPE_MAP.get(init.data_type, "UNKNOWN")
        bytes_per_elem = DTYPE_BYTES.get(dtype_name, 4)
        weight_memory += int(np.prod(tensor.shape)) * bytes_per_elem

    # Build shape inference for activation memory estimation
    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
        value_infos = {vi.name: vi for vi in inferred_model.graph.value_info}
        for inp in inferred_model.graph.input:
            value_infos[inp.name] = inp
        for out in inferred_model.graph.output:
            value_infos[out.name] = out
    except Exception:
        value_infos = {}

    # Estimate FLOPs by operation type
    flops_by_op: dict[str, int] = Counter()
    total_flops = 0

    for node in graph.node:
        flops = estimate_node_flops(node, graph, value_infos, batch_size)
        if flops > 0:
            flops_by_op[node.op_type] += flops
            total_flops += flops

    # Estimate activation memory
    activation_memory = 0
    for vi in value_infos.values():
        shape = get_tensor_shape(vi.type)
        if shape and all(isinstance(d, int) for d in shape):
            dtype_name = get_tensor_dtype(vi.type)
            bytes_per_elem = DTYPE_BYTES.get(dtype_name, 4)
            activation_memory += int(np.prod(shape)) * bytes_per_elem

    return {
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "weight_memory_bytes": weight_memory,
        "weight_memory_mb": round(weight_memory / (1024 * 1024), 2),
        "estimated_activation_memory_bytes": activation_memory,
        "estimated_activation_memory_mb": round(activation_memory / (1024 * 1024), 2),
        "total_flops": total_flops,
        "total_gflops": round(total_flops / 1e9, 4),
        "flops_by_operation": dict(
            sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True)
        ),
        "batch_size": batch_size,
    }


def estimate_node_flops(
    node, graph, value_infos: dict, batch_size: int = 1
) -> int:
    """Estimate FLOPs for a single node."""
    op_type = node.op_type

    def get_shape(name: str) -> list[int] | None:
        if name in value_infos:
            shape = get_tensor_shape(value_infos[name].type)
            if shape and all(isinstance(d, int) for d in shape):
                return [int(d) for d in shape]
        # Check initializers
        for init in graph.initializer:
            if init.name == name:
                return list(init.dims)
        return None

    def get_attr(attr_name: str, default=None):
        for attr in node.attribute:
            if attr.name == attr_name:
                if attr.type == onnx.AttributeProto.INT:
                    return attr.i
                elif attr.type == onnx.AttributeProto.INTS:
                    return list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    return attr.f
        return default

    if op_type == "Conv":
        # FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out
        weight_shape = get_shape(node.input[1]) if len(node.input) > 1 else None
        output_shape = get_shape(node.output[0]) if node.output else None

        if weight_shape and output_shape and len(weight_shape) >= 4 and len(output_shape) >= 4:
            c_out, c_in_per_group, k_h, k_w = weight_shape[:4]
            groups = get_attr("group", 1)
            c_in = c_in_per_group * groups
            h_out, w_out = output_shape[2], output_shape[3]
            flops = 2 * k_h * k_w * c_in * c_out * h_out * w_out // groups
            return flops * batch_size

    elif op_type in ("MatMul", "Gemm"):
        # FLOPs = 2 * M * N * K
        input_a_shape = get_shape(node.input[0]) if len(node.input) > 0 else None
        input_b_shape = get_shape(node.input[1]) if len(node.input) > 1 else None

        if input_a_shape and input_b_shape:
            if len(input_a_shape) >= 2 and len(input_b_shape) >= 2:
                m = input_a_shape[-2]
                k = input_a_shape[-1]
                n = input_b_shape[-1]
                flops = 2 * m * n * k
                # Account for batch dimensions
                if len(input_a_shape) > 2:
                    batch = int(np.prod(input_a_shape[:-2]))
                    flops *= batch
                return flops * batch_size

    elif op_type in ("Add", "Sub", "Mul", "Div"):
        output_shape = get_shape(node.output[0]) if node.output else None
        if output_shape:
            return int(np.prod(output_shape)) * batch_size

    elif op_type in ("Relu", "Sigmoid", "Tanh", "Softmax"):
        output_shape = get_shape(node.output[0]) if node.output else None
        if output_shape:
            # Simple activations: 1 FLOP per element (approximation)
            return int(np.prod(output_shape)) * batch_size

    elif op_type == "BatchNormalization":
        # 4 FLOPs per element (mean, var, normalize, scale)
        output_shape = get_shape(node.output[0]) if node.output else None
        if output_shape:
            return 4 * int(np.prod(output_shape)) * batch_size

    return 0


def main():
    """Run the MCP server."""
    import asyncio

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
