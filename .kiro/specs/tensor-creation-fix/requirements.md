# Requirements Document

## Introduction

This feature addresses a critical tensor creation error in the Qwen3 fine-tuning system where the training fails with "Unable to create tensor" errors related to padding and truncation issues. The system currently fails because the labels have inconsistent shapes and the data collator is not properly handling the tensor creation with appropriate padding and truncation settings.

## Requirements

### Requirement 1

**User Story:** As a machine learning engineer, I want the data collator to properly handle tensor creation with consistent shapes, so that training can proceed without tensor creation errors.

#### Acceptance Criteria

1. WHEN the data collator processes batches THEN all tensors SHALL have consistent shapes
2. WHEN labels are created THEN they SHALL be properly formatted as tensors not nested lists
3. WHEN padding is applied THEN all sequences SHALL be padded to the same length
4. WHEN truncation is needed THEN sequences SHALL be truncated to max_length
5. IF tensor creation fails THEN the system SHALL provide clear error messages with specific fixes

### Requirement 2

**User Story:** As a developer, I want the tokenization process to create properly formatted labels, so that the data collator can process them correctly.

#### Acceptance Criteria

1. WHEN tokenizing text THEN labels SHALL be created as flat lists of integers
2. WHEN copying input_ids to labels THEN the copy SHALL maintain proper tensor format
3. WHEN processing batches THEN labels SHALL have the same shape as input_ids
4. WHEN special tokens are added THEN labels SHALL account for them properly
5. IF labels have wrong format THEN the tokenization SHALL fix the format automatically

### Requirement 3

**User Story:** As a system administrator, I want robust error handling for tensor creation issues, so that training failures provide actionable debugging information.

#### Acceptance Criteria

1. WHEN tensor creation fails THEN the system SHALL log detailed shape information
2. WHEN data format is incorrect THEN the system SHALL suggest specific fixes
3. WHEN padding/truncation settings are wrong THEN the system SHALL recommend correct settings
4. WHEN debugging tensor issues THEN the system SHALL provide sample data inspection
5. IF tensor errors persist THEN the system SHALL provide fallback data processing options