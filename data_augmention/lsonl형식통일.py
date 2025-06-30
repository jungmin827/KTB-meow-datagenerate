#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
손상된 JSON 파일을 수정하고 JSONL로 변환하는 스크립트
"""

import json
import re
import sys
from pathlib import Path


def fix_json_content(content: str) -> str:
    """JSON 내용을 수정합니다."""
    
    print("🔧 JSON 내용 수정 중...")
    
    # 공백 정리
    content = content.strip()
    
    # 잘못된 이스케이프 시퀀스 수정
    content = content.replace('\\\\', '\\')
    
    # 따옴표 문제 수정
    content = re.sub(r'(?<!\\)"(?=[^,}\]])', '\\"', content)
    
    # 누락된 쉼표 추가 (간단한 패턴)
    content = re.sub(r'"}(\s*){"', '"},\n{"', content)
    
    return content


def manual_parse_json_array(content: str) -> list:
    """수동으로 JSON 배열을 파싱합니다."""
    
    print("🔧 수동 JSON 파싱 시작...")
    
    # 배열 대괄호 제거
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1].strip()
    
    # 각 JSON 객체를 분리
    objects = []
    current_obj = ""
    brace_count = 0
    in_string = False
    escape_next = False
    
    i = 0
    while i < len(content):
        char = content[i]
        
        if escape_next:
            current_obj += char
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            escape_next = True
            current_obj += char
            i += 1
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        current_obj += char
        
        # 객체가 완성되었을 때
        if not in_string and brace_count == 0 and current_obj.strip():
            try:
                # 끝의 쉼표 제거
                obj_str = current_obj.strip().rstrip(',').strip()
                if obj_str:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                    print(f"✅ 객체 {len(objects)} 파싱 성공")
            except json.JSONDecodeError as e:
                print(f"⚠️  객체 파싱 실패: {e}")
                # 더 간단한 수정 시도
                try:
                    obj_str = obj_str.replace('\\"', '"')
                    obj = json.loads(obj_str)
                    objects.append(obj)
                    print(f"✅ 객체 {len(objects)} 수정 후 파싱 성공")
                except:
                    print(f"❌ 객체 완전히 실패: {obj_str[:100]}...")
            
            current_obj = ""
        
        i += 1
    
    print(f"🔧 총 {len(objects)}개의 객체를 복구했습니다.")
    return objects


def convert_broken_json_to_jsonl(input_file: str, output_file: str) -> None:
    """손상된 JSON 파일을 JSONL로 변환합니다."""
    
    print(f"🔄 손상된 JSON 파일 변환: {input_file} → {output_file}")
    
    try:
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📖 파일 크기: {len(content)} 문자")
        
        # 먼저 표준 JSON 파싱 시도
        try:
            data = json.loads(content)
            print("✅ 표준 JSON 파싱 성공!")
        except json.JSONDecodeError as e:
            print(f"⚠️  표준 JSON 파싱 실패: {e}")
            print("🔧 수동 파싱을 시도합니다...")
            
            # JSON 내용 수정
            fixed_content = fix_json_content(content)
            
            # 수정된 내용으로 다시 시도
            try:
                data = json.loads(fixed_content)
                print("✅ 수정 후 JSON 파싱 성공!")
            except json.JSONDecodeError:
                print("🔧 수동 배열 파싱을 시도합니다...")
                data = manual_parse_json_array(content)
        
        # 리스트가 아니면 리스트로 만들기
        if not isinstance(data, list):
            data = [data]
        
        print(f"📊 총 {len(data)}개 항목을 읽었습니다.")
        
        # JSONL 파일로 저장
        valid_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                if isinstance(item, dict) and item:
                    # content 필드 확인
                    if 'content' in item and item['content'] and str(item['content']).strip():
                        json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                        f.write(json_line + '\n')
                        valid_count += 1
                        
                        if valid_count % 100 == 0:
                            print(f"📝 {valid_count}개 항목 저장됨...")
                    else:
                        print(f"⚠️  항목 {i+1}: content 없음 또는 빈 값")
                else:
                    print(f"⚠️  항목 {i+1}: 유효하지 않은 형식")
        
        print(f"✅ 변환 완료! {valid_count}개 유효한 항목이 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print("사용법: python fix_and_convert.py input.json output.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"❌ 입력 파일이 존재하지 않습니다: {input_file}")
        sys.exit(1)
    
    convert_broken_json_to_jsonl(input_file, output_file)


if __name__ == "__main__":
    main() 