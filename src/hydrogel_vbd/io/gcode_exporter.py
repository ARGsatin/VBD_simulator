from __future__ import annotations

from hydrogel_vbd.state import FieldCommand


def insert_field_commands(source_gcode: str, commands_by_layer: dict[int, FieldCommand]) -> str:
    output_lines: list[str] = []
    for line in source_gcode.splitlines():
        output_lines.append(line)
        if not line.startswith(";LAYER:"):
            continue
        layer_id = int(line.split(":", 1)[1].strip())
        command = commands_by_layer.get(layer_id)
        if command is None:
            continue
        electrode_ids = command.electrode_ids or [f"e{i}" for i in range(len(command.voltage))]
        for electrode_id, voltage in zip(electrode_ids, command.voltage):
            output_lines.append(
                f";E_FIELD: ELECTRODE={electrode_id}, VOLTAGE={float(voltage):.6f}, DURATION={command.duration:.6f}"
            )
        output_lines.append(";E_FIELD: OFF")
    return "\n".join(output_lines) + "\n"
