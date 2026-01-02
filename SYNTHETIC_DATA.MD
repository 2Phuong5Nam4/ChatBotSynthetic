🔄 Flow Đúng và Output của từng Agent

  ┌────────────────────────────────────────────────┐
  │  INPUT: raw_conversations.json + procedure     │
  └────────────────────────────────────────────────┘
                      ↓
        ┌─ Agent 1: Teacher LLM ─────────────┐
        │  INPUT: Raw conversation + Procedure│
        │  PROCESS:                           │
        │   - Analyze each turn               │
        │   - Check procedure compliance      │
        │   - Generate thinking traces        │
        │   - Propose corrections if needed   │
        │  OUTPUT:                            │
        │   - Enhanced conversation with      │
        │     thinking field in each turn     │
        └─────────────────────────────────────┘
                      ↓
        ┌─ Agent 2: Validator ───────────────┐
        │  INPUT: Enhanced conversation       │
        │         from Agent 1                │
        │  PROCESS:                           │
        │   - Step-level scoring              │
        │   - Verify thinking quality         │
        │   - Check procedure compliance      │
        │   - Flag violations/errors          │
        │  OUTPUT:                            │
        │   - Quality score (0-1)             │
        │   - Validation report               │
        │   - Accept/Reject decision          │
        └─────────────────────────────────────┘
                      ↓
           ┌─── IF score >= threshold ───┐
           │                              │
      ACCEPT │                       REJECT │
           ↓                              ↓
    ┌─ Agent 3: User Simulator ─┐   ┌─ Feedback Loop ─┐
    │  INPUT: Accepted conv      │   │  Return to       │
    │  PROCESS:                  │   │  Agent 1 with    │
    │   - Generate variants      │   │  validator       │
    │   - Create edge cases      │   │  feedback        │
    │   - Paraphrase user msgs   │   └──────────────────┘
    │  OUTPUT:                   │
    │   - 3-5 synthetic variants │
    │     per original conv      │
    └────────────────────────────┘
                  ↓
      ┌────────────────────────┐
      │  FINAL TRAINING DATA   │
      │  with thinking traces  │
      └────────────────────────┘

  ---
  📝 Ví dụ Cụ Thể với Data Thực

  INPUT (từ raw_conversations.json):

  {
    "Intentions": "C1.1.3 - Đổi mật khẩu",
    "messages": [
      {
        "role": "user",
        "content": "Em bị mất mật khẩu HVN đặt hàng, hỗ trợ em lấy lại được không?"
      },
      {
        "role": "assistant",
        "content": "Chị nhấn Quên Mật Khẩu, nhập tên đăng nhập 10409303, nhận mã OTP và tạo mật khẩu mới 12 ký tự."
      }
    ]
  }

  ---
  OUTPUT của Agent 1 (Teacher LLM):

  {
    "Intentions": "C1.1.3 - Đổi mật khẩu",
    "messages": [
      {
        "role": "user",
        "content": "Em bị mất mật khẩu HVN đặt hàng, hỗ trợ em lấy lại được không?"
      },
      {
        "role": "assistant",
        "content": "Dạ, chị vui lòng cho em xác nhận một số thông tin:\n- Chị là chủ điểm bán, NPP hay nhân viên Heineken ạ?\n- Mã cửa hàng hoặc số điện thoại đăng ký HVN của chị là gì ạ?",
        "thinking": {
          "procedure_step": "Bước 1: Nhận diện nhu cầu và xác thực sơ bộ",
          "reasoning": "Khách hàng báo mất mật khẩu, thuộc case 'Quên mật khẩu'. Theo procedure, trước khi hướng dẫn cần:\n1. Xác định đối tượng (điểm bán/NPP/nhân viên)\n2. Thu thập thông tin định danh (mã cửa hàng/NPP hoặc SĐT đăng ký)\n\nAssistant message trong raw conversation đã nhảy thẳng vào hướng dẫn với mã 10409303 mà không xác thực thông tin KH trước. Điều này vi phạm Bước 1 của procedure.",
          "correction_applied": true,
          "original_response": "Chị nhấn Quên Mật Khẩu, nhập tên đăng nhập 10409303, nhận mã OTP và tạo mật khẩu mới 12 ký tự.",
          "compliance_check": {
            "follows_procedure": false,
            "missing_steps": ["Xác định đối tượng KH", "Thu thập thông tin định danh"],
            "security_concern": "Cung cấp mã cửa hàng cụ thể (10409303) mà không xác thực danh tính KH"
          }
        }
      },
      {
        "role": "user",
        "content": "Em là chủ điểm bán, mã cửa hàng 10409303 ạ."
      },
      {
        "role": "assistant",
        "content": "Dạ, em hướng dẫn chị đặt lại mật khẩu như sau:\n\nBước 1: Trên app HVN, chị chọn 'Quên mật khẩu'\nBước 2: Nhập Tên đăng nhập là mã cửa hàng: 10409303\nBước 3: Nhập mã OTP được gửi về số điện thoại đăng ký HVN của chị\nBước 4: Tạo mật khẩu mới bao gồm 12 ký tự, lưu ý không sử dụng lại mật khẩu đã dùng trước đây ạ\nBước 5: Sau đó chị đăng nhập lại bằng mã cửa hàng và mật khẩu mới nhé",
        "thinking": {
          "procedure_step": "Bước 3: Hướng dẫn đặt lại mật khẩu qua 'Quên mật khẩu'",
          "reasoning": "Sau khi đã xác thực thông tin KH (mã cửa hàng 10409303), bây giờ hướng dẫn chi tiết các bước theo Bước 3 của procedure:\n1. Chọn Quên mật khẩu trên app\n2. Nhập tên đăng nhập (= mã cửa hàng)\n3. Nhập OTP\n4. Tạo mật khẩu mới 12 ký tự, không trùng mật khẩu cũ\n5. Đăng nhập lại\n\nCần nhấn mạnh quy tắc 12 ký tự và không dùng lại mật khẩu cũ để tránh lỗi 'mật khẩu đã sử dụng trước đó'.",
          "correction_applied": false,
          "compliance_check": {
            "follows_procedure": true,
            "covers_all_steps": true,
            "includes_password_rules": true
          }
        }
      }
    ],
    "metadata": {
      "teacher_model": "claude-3-5-sonnet",
      "corrections_made": 1,
      "procedure_compliance": "improved"
    }
  }

  ---
  OUTPUT của Agent 2 (Validator):

  {
    "conversation_id": "conv_001",
    "validation_result": {
      "overall_score": 0.92,
      "accept": true,
      "detailed_scores": {
        "procedure_compliance": 0.95,
        "thinking_quality": 0.90,
        "information_security": 0.88,
        "conversation_flow": 0.95
      },
      "turn_level_analysis": [
        {
          "turn": 1,
          "assistant_message": "Dạ, chị vui lòng cho em xác nhận...",
          "scores": {
            "procedure_step_correctness": 1.0,
            "thinking_clarity": 0.9,
            "reasoning_depth": 0.85
          },
          "validation_notes": "Correctly follows Bước 1. Thinking trace clearly explains why original response was non-compliant and what correction was needed."
        },
        {
          "turn": 2,
          "assistant_message": "Dạ, em hướng dẫn chị đặt lại mật khẩu...",
          "scores": {
            "procedure_step_correctness": 1.0,
            "thinking_clarity": 0.95,
            "reasoning_depth": 0.9
          },
          "validation_notes": "Perfect compliance with Bước 3. Thinking explains all 5 sub-steps and rationale for password rules."
        }
      ],
      "compliance_verification": {
        "all_required_steps_present": true,
        "correct_sequence": true,
        "security_guidelines_followed": true,
        "escalation_criteria_clear": true
      },
      "violations_found": [
        {
          "severity": "minor",
          "type": "incomplete_security_note",
          "description": "Thinking could mention Bước 5 về xử lý tình huống 'không nhận OTP' để coverage đầy đủ hơn",
          "recommendation": "Add a note about potential OTP issues and fallback options"
        }
      ],
      "recommendation": "ACCEPT - High quality training data with comprehensive thinking traces. Minor suggestion for improvement but not blocking."
    }
  }

  ---
  OUTPUT của Agent 3 (User Simulator):

  Dựa trên conversation đã được ACCEPT, Agent 3 tạo ra 3-5 variants với các edge cases:

  Variant 1: Edge case - KH không nhớ mã cửa hàng
  {
    "messages": [
      {
        "role": "user",
        "content": "Em quên mật khẩu HVN rồi, không biết làm sao. Em không nhớ mã cửa hàng nữa, chỉ nhớ tên quán là Nhã Vy, SĐT 0564382859."
      },
      {
        "role": "assistant",
        "content": "Dạ, anh/chị là chủ điểm bán, NPP hay nhân viên Heineken ạ? Để em tra cứu mã cửa hàng qua tên quán và số điện thoại giúp anh/chị nhé.",
        "thinking": {
          "procedure_step": "Bước 1 & Bước 2: Xác thực sơ bộ và xác định tên đăng nhập",
          "reasoning": "KH quên mật khẩu VÀ không nhớ mã cửa hàng. Theo procedure Bước 2:\n- Nếu KH không nhớ mã: tra cứu bằng tên cửa hàng + SĐT + địa chỉ\n- Gợi ý phương án đăng nhập bằng SĐT nếu phù hợp\n\nCần xác định đối tượng (Bước 1) trước khi tra cứu."
        }
      }
    ]
  }

  Variant 2: Edge case - Tài khoản bị khóa
  {
    "messages": [
      {
        "role": "user",
        "content": "App báo tài khoản em bị khóa, không đăng nhập được. Mã outlet 63235514."
      },
      {
        "role": "assistant",
        "content": "Dạ, anh/chị vui lòng gửi em ảnh màn hình báo lỗi để em xác nhận chính xác tình trạng. Với trường hợp bị khóa do nhập sai mật khẩu nhiều lần, anh/chị có thể mở khóa bằng cách thực hiện 'Quên mật khẩu' để đặt lại mật khẩu mới ạ.",
        "thinking": {
          "procedure_step": "Bước 1 & Bước 5: Nhận diện + xử lý tình huống bị khóa",
          "reasoning": "KH báo tài khoản bị khóa - thuộc tình huống xử lý ở Bước 5 của procedure:\n- Tài khoản bị khóa: Thực hiện Bước 3 (Quên mật khẩu) để mở khóa\n\nNên yêu cầu ảnh màn hình để xác định đúng loại lỗi trước khi hướng dẫn."
        }
      }
    ]
  }

  Variant 3: Paraphrase - Cách hỏi khác
  {
    "messages": [
      {
        "role": "user",
        "content": "Anh ơi, sáng nãy anh vào app HVN được, giờ nó báo sai mật khẩu mãi. Có lẽ anh nhập sai quá nhiều lần rồi."
      },
      // ... tương tự nhưng với tone và context khác
    ]
  }

  ---
  🎯 Tóm lại ý nghĩa của từng Agent:

  | Agent                   | Input                            | Output                                            | Mục đích                                              |
  |-------------------------|----------------------------------|---------------------------------------------------|-------------------------------------------------------|
  | Agent 1: Teacher LLM    | Raw conversation + Procedure     | Enhanced conversation với thinking traces         | Tạo training data chất lượng cao, tuân thủ procedure  |
  | Agent 2: Validator      | Enhanced conversation từ Agent 1 | Quality score + Validation report + Accept/Reject | Quality control - chỉ cho qua training data đạt chuẩn |
  | Agent 3: User Simulator | Accepted conversations           | 3-5 variants với edge cases                       | Data augmentation - tăng diversity và coverage        |

  Validation report của Agent 2 có ý nghĩa:
  - ✅ Quality gate: Chỉ training data đạt ngưỡng mới được dùng
  - ✅ Feedback loop: Data bị reject được gửi lại Agent 1 với chi tiết lỗi để sửa
  - ✅ Metrics tracking: Theo dõi chất lượng tổng thể của synthetic data pipeline

  Bạn có muốn tôi giải thích thêm phần nào không?