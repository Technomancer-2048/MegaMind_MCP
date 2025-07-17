# Database Schema Mapping Reference

## Column Mapping for SQL Query Fixes

### megamind_sessions Table
| My Implementation | Actual Database | Type | Notes |
|------------------|----------------|------|-------|
| `session_type` | `session_state` | enum('open','active','archived') | State instead of type |
| `created_by` | `user_id` | varchar(100) | User ID instead of created_by |
| `description` | `session_name` | varchar(255) | Name instead of description |
| `status` | `session_state` | enum('open','active','archived') | Same as session_type |
| `metadata` | `session_config` | json | Config instead of metadata |
| `last_activity` | `last_activity` | timestamp | ✅ Matches |
| `created_at` | `created_at` | timestamp | ✅ Matches |

### megamind_promotion_queue Table
| My Implementation | Actual Database | Type | Notes |
|------------------|----------------|------|-------|
| `chunk_id` | `source_chunk_id` | varchar(50) | Source chunk ID |
| `source_realm` | `source_realm_id` | varchar(50) | Realm ID suffix |
| `target_realm` | `target_realm_id` | varchar(50) | Realm ID suffix |
| `created_by` | `requested_by` | varchar(100) | Requested by user |
| `created_date` | `requested_at` | timestamp | Requested at timestamp |
| `justification` | `justification` | text | ✅ Matches |
| `status` | `status` | enum('pending','approved','rejected','processing','completed') | ✅ Matches |

### megamind_session_changes Table
| My Implementation | Actual Database | Type | Notes |
|------------------|----------------|------|-------|
| `change_type` | `change_type` | enum('create_chunk','update_chunk','add_relationship','add_tag') | ✅ Matches |
| `change_details` | `change_data` | json | Data instead of details |
| `created_at` | `created_at` | timestamp | ✅ Matches |
| `status` | `status` | enum('pending','approved','rejected','applied') | ✅ Matches |

### megamind_promotion_history Table
| My Implementation | Actual Database | Type | Notes |
|------------------|----------------|------|-------|
| `action` | `action_type` | enum('created','approved','rejected','completed','failed','cancelled') | Action type |
| `action_reason` | `action_reason` | text | ✅ Matches |
| `action_by` | `action_by` | varchar(100) | ✅ Matches |
| `action_date` | `action_at` | timestamp | At instead of date |

### megamind_user_feedback Table
| My Implementation | Actual Database | Type | Notes |
|------------------|----------------|------|-------|
| `feedback_data` | `details` | json | Details instead of data |
| `feedback_type` | `feedback_type` | enum('chunk_quality','boundary_accuracy','retrieval_success','manual_correction') | ✅ Matches |
| `created_at` | `created_date` | timestamp | Date instead of at |
| `session_id` | `session_id` | varchar(50) | ✅ Matches |

## Additional Database Fields Available

### megamind_sessions - Additional Fields
- `realm_id` - varchar(50) - Current realm
- `project_context` - varchar(255) - Project context
- `session_tags` - json - Session tags
- `priority` - enum('low','medium','high','critical') - Session priority
- `enable_semantic_indexing` - tinyint(1) - Semantic indexing enabled
- `content_token_limit` - int - Token limit
- `embedding_generation_enabled` - tinyint(1) - Embedding generation enabled
- `total_entries` - int - Total entries count
- `total_chunks_accessed` - int - Chunks accessed count
- `total_operations` - int - Operations count
- `performance_score` - decimal(3,2) - Performance score
- `context_quality_score` - decimal(3,2) - Quality score

### megamind_promotion_queue - Additional Fields
- `promotion_type` - enum('copy','move','reference') - Promotion type
- `reviewed_by` - varchar(100) - Reviewer
- `reviewed_at` - timestamp - Review timestamp
- `completed_at` - timestamp - Completion timestamp
- `business_impact` - enum('low','medium','high','critical') - Business impact
- `review_notes` - text - Review notes
- `original_content` - text - Original content
- `target_chunk_id` - varchar(50) - Target chunk ID
- `promotion_session_id` - varchar(50) - Promotion session ID

### megamind_session_changes - Additional Fields
- `target_chunk_id` - varchar(50) - Target chunk ID
- `source_realm_id` - varchar(50) - Source realm ID
- `impact_score` - decimal(3,2) - Impact score
- `priority` - enum('low','medium','high','critical') - Change priority

### megamind_user_feedback - Additional Fields
- `target_id` - varchar(50) - Target ID
- `rating` - float - Rating value
- `user_id` - varchar(100) - User ID