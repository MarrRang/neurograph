from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner


@dataclass(frozen=True)
class V01Project:
    root: Path
    server: Path
    signup_form: Path
    signup_schema: Path
    status_enum: Path
    openapi: Path
    markdown_spec: Path


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sb_page_text() -> str:
    return "\n".join(
        [
            "회원가입 화면",
            "회원은 이메일과 password 필드를 입력한다.",
            "비밀번호는 8자 이상이어야 한다.",
            "POST /users/signup 호출 후 status 값은 PENDING 또는 APPROVED 또는 FAILED 중 하나다.",
            '실패 시 오류 메시지: "비밀번호는 8자 이상이어야 합니다"',
        ]
    )


@pytest.fixture
def small_ts_api_project(tmp_path: Path) -> V01Project:
    src = tmp_path / "src"
    docs = tmp_path / "docs"
    src.mkdir()
    docs.mkdir()

    server = src / "server.ts"
    server.write_text(
        "import express from 'express';\n"
        "import { signupSchema } from './signupSchema';\n"
        "import { SignupStatus } from './status';\n"
        "const app = express();\n"
        "function signupHandler(req, res) {\n"
        "  const parsed = signupSchema.parse(req.body);\n"
        "  return res.status(201).json({ status: SignupStatus.PENDING, userId: parsed.email });\n"
        "}\n"
        "app.post('/users/signup', signupHandler);\n",
        encoding="utf-8",
    )

    signup_form = src / "SignupForm.tsx"
    signup_form.write_text(
        "export function SignupForm() {\n"
        "  return <form action=\"/users/signup\"><input name=\"email\" /><input name=\"password\" /></form>;\n"
        "}\n",
        encoding="utf-8",
    )

    signup_schema = src / "signupSchema.ts"
    signup_schema.write_text(
        "import { z } from 'zod';\n"
        "export const signupSchema = z.object({\n"
        "  email: z.string().email(),\n"
        "  password: z.string().min(8),\n"
        "});\n"
        "export function validateSignup(input: unknown) {\n"
        "  return signupSchema.parse(input);\n"
        "}\n",
        encoding="utf-8",
    )

    status_enum = src / "status.ts"
    status_enum.write_text(
        "export enum SignupStatus {\n"
        "  PENDING = 'PENDING',\n"
        "  APPROVED = 'APPROVED',\n"
        "  FAILED = 'FAILED',\n"
        "}\n",
        encoding="utf-8",
    )

    openapi = tmp_path / "openapi.yaml"
    openapi.write_text(
        "openapi: 3.1.0\n"
        "paths:\n"
        "  /users/signup:\n"
        "    post:\n"
        "      operationId: signupHandler\n"
        "components:\n"
        "  schemas:\n"
        "    SignupResponse:\n"
        "      properties:\n"
        "        status:\n"
        "          enum:\n"
        "            - PENDING\n"
        "            - APPROVED\n"
        "            - FAILED\n",
        encoding="utf-8",
    )

    markdown_spec = docs / "signup-spec.md"
    markdown_spec.write_text(
        "# Signup Spec\n\n"
        "The web client calls POST /users/signup for 회원가입.\n\n"
        "The `password` field must be 8자 이상 and email is required.\n\n"
        'Show error message `"비밀번호는 8자 이상이어야 합니다"` when validation fails.\n',
        encoding="utf-8",
    )

    return V01Project(
        root=tmp_path,
        server=server,
        signup_form=signup_form,
        signup_schema=signup_schema,
        status_enum=status_enum,
        openapi=openapi,
        markdown_spec=markdown_spec,
    )


@pytest.fixture
def conflict_fixture(tmp_path: Path) -> Path:
    (tmp_path / "signup.md").write_text(
        "# Signup\n\n비밀번호는 8자 이상이어야 한다.\n",
        encoding="utf-8",
    )
    (tmp_path / "signupSchema.ts").write_text(
        "export const signupSchema = z.object({\n"
        "  password: z.string().min(6),\n"
        "});\n",
        encoding="utf-8",
    )
    return tmp_path
