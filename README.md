# onnx-mcp

ONNX 모델 구조와 메타데이터를 분석하는 MCP 서버입니다.

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/jhkimnota/onnx-mcp.git
cd onnx-mcp
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -e .
```

설치가 완료되면 `onnx-mcp-server` 명령어를 사용할 수 있습니다.

## MCP 클라이언트 설정

### Claude Code

`~/.config/claude/claude_desktop_config.json` 파일을 열고 다음 내용을 추가합니다:

```json
{
  "mcpServers": {
    "onnx": {
      "command": "/path/to/onnx-mcp/venv/bin/onnx-mcp-server"
    }
  }
}
```

> `/path/to/onnx-mcp`는 실제 클론한 경로로 변경하세요.

### Cursor

Cursor 설정에서 MCP 서버를 추가합니다. 설정 파일 위치는 OS에 따라 다릅니다:

- macOS: `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Linux: `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Windows: `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "onnx": {
      "command": "/path/to/onnx-mcp/venv/bin/onnx-mcp-server"
    }
  }
}
```

설정 후 Claude Code 또는 Cursor를 재시작하면 ONNX 관련 도구를 사용할 수 있습니다.

## 제공 기능

- **get_model_info**: IR 버전, opset, producer 등 모델 메타데이터 조회
- **get_inputs_outputs**: 입출력 텐서의 이름, 데이터 타입, shape 정보
- **get_graph_structure**: 노드 목록, 연결 관계, 연산 타입별 통계
- **get_node_details**: 특정 노드의 속성, 입출력, 연산 타입 상세 정보
- **visualize_graph**: Mermaid 다이어그램으로 그래프 구조 시각화
- **get_weight_statistics**: 가중치별 shape, 메모리 사용량, 값의 분포 통계
- **estimate_model_complexity**: 메모리 사용량 및 FLOPs 추정

## 사용 예시

MCP가 연결된 Claude Code 또는 Cursor에서 자연어로 질문하면 자동으로 적절한 도구를 사용합니다:

```
이 ONNX 모델의 입력 shape이 어떻게 돼?

모델에서 사용된 연산 타입별로 개수를 알려줘

첫 번째 Conv 노드의 상세 속성을 보여줘

이 모델의 전체 파라미터 개수와 메모리 사용량은?

배치 사이즈 1일 때 FLOPs가 얼마나 나와?

그래프 구조를 다이어그램으로 보여줄 수 있어?

모델의 opset 버전이 뭐야?
```
