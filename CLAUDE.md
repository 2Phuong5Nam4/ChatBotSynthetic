# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chatbot synthetic data generation project for Heineken Vietnam's ordering system (HVN Đặt Hàng). The repository contains structured procedural prompts and example conversations in Vietnamese for training customer support chatbots across multiple channels (HVN app, SEM, DIS Lite).

## Commands

### Setup
```bash
# The project uses Python 3.13 and UV for dependency management
# Create and activate virtual environment (if needed)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv (recommended)
uv sync

# Or add new dependencies
uv add <package-name>
```

### Running
```bash
# Run the main script
python main.py

# Or using uv (recommended for this project)
uv run main.py
```

### Testing
```bash
# This project uses pytest for testing
# Run all tests
uv run python -m pytest

# Run specific test file
uv run python -m pytest tests/test_dataset_loader.py

# Run with verbose output
uv run python -m pytest -v

# Run specific test function
uv run python -m pytest tests/test_dataset_loader.py::TestDatasetLoader::test_load_train_dataset

# Run tests excluding slow ones
uv run python -m pytest -m "not slow"
```

## Architecture

### Data Pipeline and Processing Flow

This project follows a three-stage data processing pipeline:

```
Chat Script_For Chatbot Demo.xlsx (Source)
              ↓
         [Analysis & Extraction]
              ↓
    prompts/*.json (Structured Procedures)
              ↓
      [Standardization & Enrichment]
              ↓
    raw_conversations.json (Training Data)
```

### Data Structure

1. **Source Data** (`Chat Script_For Chatbot Demo.xlsx`):
   - **Origin**: Original conversation scripts provided by the user/business team
   - **Content**: Raw chatbot demonstration scripts and conversation flows
   - **Purpose**: Primary source material containing business requirements and conversation examples

2. **Extracted Procedure Files** (Generated from Excel):

   **Master File** - `procedure.json`:
   - Contains 5 main procedures extracted and structured from the Excel source
   - Numbered procedures (1, 2, 3, 4, 5) covering core support scenarios
   - Each procedure has:
     - `detail_description`: Comprehensive step-by-step Vietnamese instructions (Bước 1, Bước 2, etc.)
     - `example_conversation`: Dictionary of reference conversations (conversation_1, conversation_2, etc.)

   **Individual Prompt Files** (Extracted from procedure.json):
   - `build_login_prompt.json`: Login support procedures (Procedure 1)
   - `forget_password_prompt.json`: Password recovery workflows (Procedure 2)
   - `check_order_prompt.json`: Order verification processes (Procedure 3)
   - `check_outlet_npp_relationship_prompt.json`: Outlet-NPP relationship validation (Procedure 4)
   - `order_guidance_prompt.json`: Order placement guidance (Procedure 5)
   - Each includes additional `short_description` field for quick reference

3. **Standardized Training Data** (`raw_conversations.json`):
   - **Origin**: Standardized and normalized version of the Excel source
   - **Extraction**: Structured metadata for chatbot training:
     - `cations`: Application channel (HVN, SEM, DIS Lite)
     - `Category`: Main functional category (e.g., "C1 - Online Portal")
     - `Sub_Category`: Detailed classification (e.g., "C1.1 - User Management")
     - `Targeted_Customers`: Customer segments (Outlet, Sales Force, Distributor/Sub-Distributor)
     - `Intentions`: Intent codes for classification (e.g., "C1.1.3 - Đổi mật khẩu")
     - `Solutions`: Condensed solution steps (simplified from `detail_description`)
     - `messages`: Multi-turn conversations in user/assistant format
   - **Purpose**: Ready-to-use training data with intent classification and routing metadata

### Key Domain Concepts

- **HVN (Heineken Vietnam)**: Main customer-facing ordering application
- **SEM**: Sales Force Execution Mobile application
- **DIS Lite**: Distributor Lite Portal (Seller Portal)
- **Outlet/OutletID**: Retail store identification (8-digit codes like 63235514)
- **NPP**: Nhà phân phối (Distributor)
- **SubD/SDIP**: Sub-Distributor
- **MQH**: Mối quan hệ (Relationship) - connections between outlets and distributors
- **Gratis**: Free promotional orders
- **OTP**: One-time password for authentication
- **SA**: Sale Admin

### Data Processing Relationships

The relationship between files follows this transformation flow:

1. **Excel → Procedures**: `Chat Script_For Chatbot Demo.xlsx` is analyzed to extract business logic into structured `procedure.json`
2. **Procedures → Individual Prompts**: Each procedure in `procedure.json` is split into separate prompt files for modular use
3. **Excel → raw_conversations**: The original Excel conversations are standardized into `raw_conversations.json` with added metadata taxonomy

**Important Notes**:
- The prompt files serve as **procedure documentation** for how to handle scenarios
- The raw_conversations file contains **actual training conversations** with intent classification
- Metadata fields (Category, Sub_Category, Intentions) in raw_conversations are the original classifications from the Excel source
- Both derive from the same Excel source but serve different purposes (procedures vs. training data)

## Working with Prompts

### Prompt File Structure

When reading or modifying prompt files:

1. **Procedure Format**:
   - Vietnamese text with step-by-step instructions (Bước 1, Bước 2, etc.)
   - Reference conversations by ID (e.g., "xem conversation_1")
   - Include business rules, validation steps, and escalation criteria

2. **Conversation Format**:
   - ID-based conversations in nested dictionaries
   - Each conversation has `id` and `content` with realistic user-agent exchanges
   - Content uses Vietnamese language with domain-specific terminology

3. **Raw Conversations Format**:
   - Array of conversation objects
   - Each has metadata (category, intentions, solutions) and message arrays
   - Messages alternate between user and assistant roles

### Common Scenarios Covered

1. **User Management** (C1.1):
   - Login support (C5.1.1)
   - Password reset (C1.1.3)
   - Account activation/deactivation

2. **Order Taking** (C2.2):
   - Checking orders (C2.2.1)
   - Order taking guidance (C2.2.3)
   - Missing orders from platforms (C2.1.1)

3. **Relationship Management** (C5.5):
   - Outlet-NPP/SubD relationship verification (C5.5.1)
   - Relationship validity checks
   - MQH troubleshooting

4. **Order Status** (C2.4):
   - Sale order checking (C2.4.1)
   - Order synchronization issues

## Development Patterns

### Vietnamese Language

- All user-facing content is in Vietnamese
- Maintain formal Vietnamese customer service tone (use of "anh/chị", "em", polite particles "ạ")
- Preserve domain-specific acronyms (NPP, SubD, MQH, etc.)

### ID and Code Formats

- OutletID: 8 digits (e.g., 67803609)
- NPP/SubD codes: 8 digits (e.g., 10375694)
- Order codes: Various formats (e.g., 2509076469100, CO251124-01481)
- Phone numbers: 10 digits starting with 0 (e.g., 0327592751)

### Common Workflows

1. **Identity Verification**: Always collect and verify OutletID, store name, and phone number
2. **Relationship Validation**: Check MQH between outlets and distributors before processing orders
3. **24-hour Sync Rule**: After relationship changes, wait 24 hours for system synchronization
4. **Escalation Paths**: Route to Sale Admin, Sales Team, or hotline 19001845 as appropriate

### Data Integrity

- Conversations reference specific outlet codes and scenarios
- Maintain consistency between procedure descriptions and example conversations
- Keep security guidelines (never share passwords directly, use OTP recovery)

