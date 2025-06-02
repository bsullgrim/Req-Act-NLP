# Enhanced Dependency Analysis Workflow Guide

## Overview
This package contains automated dependency analysis results with detailed explanations for each match decision.

## File Structure
- `dependency_review_workbook_explained.xlsx`: Main review workbook with explanations
- `matches_with_explanations.json`: Machine-readable data with full explanations
- `action_items_explained.csv`: Project management tracking with explanation summaries
- `explanation_summary.html`: Visual summary report with top matches explained
- `match_summary_enhanced.json`: Dashboard data with explanation metrics

## Enhanced Review Process

### 1. Auto-Approve Sheet
- **High confidence matches (>70%) with strong semantic similarity**
- **Key indicators**: "Very High" or "High" semantic similarity, multiple shared terms
- **Explanation focus**: Verify the semantic explanation makes sense
- **Estimated time**: 15 minutes per match

### 2. Quick Review Sheet
- **High confidence OR medium confidence with good semantic match**
- **Key indicators**: Moderate to high semantic similarity, some shared technical terms
- **Explanation focus**: Check BM25 and domain explanations for technical alignment
- **Estimated time**: 30 minutes per match

### 3. Detailed Review Sheet
- **Medium confidence matches requiring thorough analysis**
- **Key indicators**: Mixed signals across different scoring components
- **Explanation focus**: Analyze all explanation components, especially syntactic patterns
- **Estimated time**: 2 hours per match

### 4. Manual Analysis Sheet
- **Low confidence matches that may be false positives**
- **Key indicators**: Low semantic similarity, few shared terms, weak explanations
- **Explanation focus**: Determine if match has any validity despite low scores
- **Estimated time**: 4 hours per match

## Explanation Components Guide

### Semantic Explanation
- **Very High (>0.7)**: Strong conceptual match, likely correct dependency
- **High (0.5-0.7)**: Good conceptual alignment, review context
- **Medium (0.3-0.5)**: Moderate similarity, may need domain expert review
- **Low (<0.3)**: Weak conceptual connection, likely false positive

### BM25 Explanation
- Shows exact term matches and their statistical importance
- Look for meaningful technical terms, not just common words
- Higher scores with technical terms indicate stronger matches

### Domain Explanation
- Highlights shared technical terminology
- Focus on domain-specific terms with high weights
- Empty explanations may indicate vocabulary mismatch

### Syntactic Explanation
- Compares sentence structure and linguistic patterns
- Useful for identifying functionally similar activities
- Higher scores suggest similar action patterns

### Query Expansion Explanation
- Shows synonym and related term matching
- Helps identify matches despite different terminology
- Review expanded terms for domain appropriateness

## Using the Explanation Data

### In Excel Workbook
1. **Key Evidence column**: Quick summary of main match indicators
2. **Explanation columns**: Detailed breakdowns for each scoring component
3. **Shared Terms column**: Direct vocabulary overlap
4. **Match Quality**: Overall assessment (EXCELLENT/GOOD/MODERATE/WEAK)

### In JSON Data
- Full programmatic access to all explanations
- Suitable for integration with other tools
- Contains structured evidence and reasoning

### In HTML Report
- Visual overview of top matches
- Color-coded by confidence level
- Easy sharing with stakeholders

## Review Workflow

1. **Start with Excel workbook** - gives best overview and input capability
2. **Use explanation columns** to understand why each match was suggested
3. **Cross-reference with HTML report** for visual confirmation of top matches
4. **Update status and notes** directly in Excel
5. **Export decisions** using action items CSV for project tracking

## Quality Indicators

### Strong Match Indicators
- Multiple shared technical terms
- High semantic similarity with clear explanation
- Domain-specific term overlap
- Consistent scores across components

### Weak Match Indicators
- No shared meaningful terms
- Low semantic similarity
- Generic or common-word overlap only
- Inconsistent scores across components

## Decision Guidelines

### Approve if:
- Semantic explanation makes logical sense
- Multiple technical terms are shared
- Domain explanation shows relevant technical overlap
- Match quality is EXCELLENT or GOOD

### Review Further if:
- Mixed signals across explanation components
- Semantic similarity is moderate but term overlap is low
- Domain explanation is weak but other indicators are strong
- Match quality is MODERATE

### Reject if:
- No meaningful shared technical terms
- Semantic explanation shows conceptual mismatch
- All explanation components are weak
- Match quality is WEAK with no redeeming factors

## Integration with Engineering Process

1. **Requirements Traceability**: Use approved matches for formal traceability matrix
2. **Architecture Review**: Reference explanations in design review meetings  
3. **Test Planning**: Use matched activities for test case derivation
4. **Documentation**: Include match reasoning in requirement specifications
5. **Change Impact**: Re-run analysis when requirements change

## Troubleshooting

### If explanations seem unclear:
- Check the original requirement and activity texts for context
- Review the explanation guide sheet in Excel workbook
- Consult domain experts for technical term validation

### If scores seem inconsistent:
- Different components measure different aspects of similarity
- Low semantic with high BM25 may indicate technical term match without conceptual alignment
- High semantic with low BM25 may indicate conceptual match with different terminology

### If no good matches found:
- Requirement may be too abstract or implementation-specific
- Activity descriptions may need more detail
- Consider if requirement should be decomposed

## Support

For questions about the matching algorithm or explanations, refer to:
- Explanation Guide sheet in Excel workbook
- Technical documentation in algorithm source code
- Domain expert consultation for technical term validation
