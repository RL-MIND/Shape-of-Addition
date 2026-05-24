"""
Interactive UMAP HTML visualization module.

Uses Plotly to generate interactive HTML files with hover tooltips showing Q, GT, Pred,
and bolds the digit at the current position.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def bold_digit_at_pos(text: str, pos: int, is_expression: bool = False) -> str:
    """
    Bold the digit at the specified position in the text.
    
    Args:
        text: Input text
        pos: Position index (0-based, 0 is the most significant digit)
        is_expression: Whether the text is an expression (e.g. "12 + 34"); each number is handled separately
        
    Returns:
        Text with HTML bold tags
    """
    if is_expression:
        # For expressions, find all numbers and process each separately
        # e.g. "12 + 34" -> extract ["12", "34"]
        parts = re.split(r'(\d+)', text)
        result_parts = []
        for part in parts:
            if part.isdigit():
                # Bold the digit at position pos in each number
                if pos < len(part):
                    bolded = part[:pos] + f"<b>{part[pos]}</b>" + part[pos+1:]
                    result_parts.append(bolded)
                else:
                    result_parts.append(part)
            else:
                result_parts.append(part)
        return ''.join(result_parts)
    else:
        # For a single number (GT or Pred), bold the digit at position pos directly
        # Strip possible whitespace
        text = text.strip()
        if pos < len(text) and text[pos].isdigit():
            return text[:pos] + f"<b>{text[pos]}</b>" + text[pos+1:]
        return text


def generate_hover_text(
    sample_indices: np.ndarray,
    samples_meta: Dict[int, Dict[str, str]],
    position: Union[int, str],
    labels: np.ndarray
) -> List[str]:
    """
    Generate the list of hover tooltip texts.
    
    Args:
        sample_indices: Array of sample indices
        samples_meta: Sample metadata loaded from h5 {sample_idx: {"question": ..., "pred": ..., "gt": ...}}
        position: Current position (int or string such as 'extra')
        labels: Label array (used to show correct/incorrect status)
        
    Returns:
        List of hover texts in HTML format
    """
    hover_texts = []
    pos = int(position) if isinstance(position, (int, np.integer)) else None
    
    for i, sample_idx in enumerate(sample_indices):
        sample_idx = int(sample_idx)
        meta = samples_meta.get(sample_idx, {})
        
        question = meta.get("question", "N/A")
        gt = meta.get("gt", "N/A")
        pred = meta.get("pred", "N/A")
        
        # Clean pred (may contain extra characters)
        # Keep only digit characters
        pred_clean = ''.join(c for c in pred if c.isdigit() or c == '-')
        if not pred_clean:
            pred_clean = pred  # If empty after cleaning, keep original value
        
        if pos is not None and pos >= 0:
            # Bold the digit at the current pos
            question_display = bold_digit_at_pos(question, pos, is_expression=True)
            gt_display = bold_digit_at_pos(gt, pos, is_expression=False)
            pred_display = bold_digit_at_pos(pred_clean, pos, is_expression=False)
        else:
            question_display = question
            gt_display = gt
            pred_display = pred_clean
        
        # Add correct/incorrect marker
        status = "✓" if labels[i] == 1 else "✗"
        
        hover_text = (
            f"Q: {question_display}<br>"
            f"GT: {gt_display}<br>"
            f"Pred: {pred_display}<br>"
            f"Status: {status}"
        )
        hover_texts.append(hover_text)
    
    return hover_texts


def create_interactive_html(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    samples_meta: Dict[int, Dict[str, str]],
    position: Union[int, str],
    layer_idx: int,
    save_dir: str,
    umap_dim: int = 2,
    correctness: Optional[np.ndarray] = None,
    color_labels: Optional[np.ndarray] = None,
    color_mode: str = "consistent",
    marker_mode: str = "point",
    digits: Optional[np.ndarray] = None,
    gt_digits: Optional[np.ndarray] = None,
    pred_digits: Optional[np.ndarray] = None,
    in_carry_digits: Optional[np.ndarray] = None,
    t_inertia_digits: Optional[np.ndarray] = None,
    pred_filter_suffix: str = "",
    anchor_embedding: Optional[np.ndarray] = None,
    anchor_labels: Optional[np.ndarray] = None,
    grid_tokens: Optional[list] = None
) -> str:
    """
    Create an interactive UMAP HTML visualization.
    
    Args:
        embedding: UMAP embedding coordinates (N, 2) or (N, 3)
        labels: Binary labels used for marker logic (N,)
        color_labels: Optional binary labels used for color (N,). Defaults to labels if None
        sample_indices: Sample indices (N,)
        samples_meta: Sample metadata dict
        position: Current position
        layer_idx: Current layer index
        save_dir: Output directory
        umap_dim: UMAP dimension (2 or 3)
        correctness: Model prediction correctness array (optional, highlights wrong predictions)
        color_mode: Color mode ("consistent" or "prefix_is_correct")
        marker_mode: Marker mode ("point", "pred", "input", "in_carry", "out_carry", "raw_sum", "gt_pred_carry", "gt_pred_carry_Inertia")
        digits: Digit array for display (when marker_mode is not "point")
        gt_digits: GT digit array (for negative samples showing pred(gt) format)
        pred_digits: Pred digit array (for in_carry/out_carry modes showing carry(pred) format)
        in_carry_digits: Input carry array (for raw_sum mode showing raw_sum(in_carry) format)
        t_inertia_digits: T_inertia values (for gt_pred_carry_Inertia mode)
        
    Returns:
        Path to the saved HTML file
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Run: pip install plotly")
    
    # Generate hover texts
    labels = np.asarray(labels)
    labels_for_color = labels if color_labels is None else np.asarray(color_labels)
    if len(labels_for_color) != len(labels):
        raise ValueError(
            f"color_labels length ({len(labels_for_color)}) does not match labels length ({len(labels)})"
        )
    hover_texts = generate_hover_text(sample_indices, samples_meta, position, labels_for_color)
    
    # Define colors
    colors = []
    for i, label in enumerate(labels_for_color):
        if correctness is not None and not correctness[i]:
            colors.append('green')  # Wrong model prediction in green
        elif label == 1:
            colors.append('blue')   # Positive sample (GT correct)
        else:
            colors.append('red')    # Negative sample (GT incorrect)
    
    # Use point markers or text markers depending on marker_mode
    use_text_marker = marker_mode != "point" and digits is not None
    
    # Generate text markers if needed
    text_markers = None
    if use_text_marker:
        text_markers = []
        for i, (d, label) in enumerate(zip(digits, labels)):
            if d >= 0:
                if marker_mode in ["in_carry", "out_carry"]:
                    # in_carry/out_carry mode: show carry(pred)
                    if pred_digits is not None and pred_digits[i] >= 0:
                        text_markers.append(f"{int(d)}({int(pred_digits[i])})")
                    else:
                        text_markers.append(str(int(d)))
                elif marker_mode == "raw_sum":
                    # raw_sum mode: show raw_sum(in_carry)
                    if in_carry_digits is not None and in_carry_digits[i] >= 0:
                        text_markers.append(f"{int(d)}({int(in_carry_digits[i])})")
                    else:
                        text_markers.append(str(int(d)))
                elif marker_mode == "gt_pred_carry":
                    # gt_pred_carry mode: show (gt, pred, in_carry)
                    p = int(pred_digits[i]) if pred_digits is not None else -1
                    ic = int(in_carry_digits[i]) if in_carry_digits is not None else -1
                    text_markers.append(f"({int(gt_digits[i])},{p},{ic})")
                elif marker_mode in ["gt_pred_carry_potential"]:
                    # gt_pred_carry_Inertia/potential mode: show (gt, pred, in_carry, c_potential)
                    p = int(pred_digits[i]) if pred_digits is not None else -1
                    ic = int(in_carry_digits[i]) if in_carry_digits is not None else -1
                    ti = float(t_inertia_digits[i]) if t_inertia_digits is not None else 0.0
                    text_markers.append(f"({int(gt_digits[i])},{p},{ic},{ti:.2f})")
                elif label == 0 and gt_digits is not None and gt_digits[i] >= 0:
                    # Negative samples show digit(gt) format (pred/input modes only)
                    if marker_mode in ["pred", "input"]:
                        text_markers.append(f"{int(d)}({int(gt_digits[i])})")
                    else:
                        text_markers.append(str(int(d)))
                else:
                    text_markers.append(str(int(d)))
            elif d == -2:  # PL marker
                if label == 0 and gt_digits is not None and gt_digits[i] >= 0:
                    text_markers.append(f"PL({int(gt_digits[i])})")
                else:
                    text_markers.append("PL")
            else:
                text_markers.append("")  # Do not display invalid values
    
    # Group data by color so each group has its own hover background color
    color_groups = {'blue': [], 'red': [], 'green': []}
    for i, color in enumerate(colors):
        color_groups[color].append(i)
    
    # Create figure
    fig = go.Figure()
    
    if color_mode == "prefix_is_correct":
        blue_label = "Prefix Correct"
        red_label = "Prefix Incorrect"
    else:
        blue_label = "Positive (correct)"
        red_label = "Negative (incorrect)"

    color_labels = {
        'blue': blue_label,
        'red': red_label,
        'green': 'Wrong prediction'
    }
    
    for color, indices in color_groups.items():
        if not indices:
            continue
            
        indices = np.array(indices)
        x_data = embedding[indices, 0]
        y_data = embedding[indices, 1]
        z_data = embedding[indices, 2] if (umap_dim == 3 and embedding.shape[1] >= 3) else None
        hover_data = [hover_texts[i] for i in indices]
        
        if use_text_marker:
            text_data = [text_markers[i] for i in indices]
            if umap_dim == 3 and embedding.shape[1] >= 3:
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='text',
                    text=text_data,
                    textfont=dict(size=8, color=color),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_data,
                    hoverlabel=dict(bgcolor=color, font_color='white'),
                    name=color_labels[color],
                    showlegend=True
                )
            else:
                trace = go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='text',
                    text=text_data,
                    textfont=dict(size=10, color=color),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_data,
                    hoverlabel=dict(bgcolor=color, font_color='white'),
                    name=color_labels[color],
                    showlegend=True
                )
        else:
            if umap_dim == 3 and embedding.shape[1] >= 3:
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.6),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_data,
                    hoverlabel=dict(bgcolor=color, font_color='white'),
                    name=color_labels[color],
                    showlegend=True
                )
            else:
                trace = go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(size=6, color=color, opacity=0.6),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_data,
                    hoverlabel=dict(bgcolor=color, font_color='white'),
                    name=color_labels[color],
                    showlegend=True
                )
        fig.add_trace(trace)
    
    # Set layout
    if umap_dim == 3 and embedding.shape[1] >= 3:
        fig.update_layout(
            title=f'UMAP 3D - Position {position}, Layer {layer_idx} [marker: {marker_mode}]',
            scene=dict(
                xaxis_title='UMAP-1',
                yaxis_title='UMAP-2',
                zaxis_title='UMAP-3'
            ),
            autosize=True  # Auto-resize
        )
    else:
        fig.update_layout(
            title=f'UMAP 2D - Position {position}, Layer {layer_idx} [marker: {marker_mode}]',
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            autosize=True,  # Auto-resize
            dragmode='pan'  # Default drag mode is pan
        )

    
    # Add Grid Decoded Regions (Scatter/Scatter3d grouped by token for color and legend)
    if grid_tokens is not None:
        grid_tokens_filtered = [item for item in grid_tokens if item[-1]] # Filter empty tokens
        
        if grid_tokens_filtered:
            import plotly.colors as pcolors
            from itertools import cycle
            
            # Extract unique tokens and assign colors
            all_tokens = [item[-1] for item in grid_tokens_filtered]
            unique_tokens = sorted(list(set(all_tokens)))
            # Use Plotly default color cycle
            color_cycle = cycle(pcolors.qualitative.Plotly)
            token_to_color = {t: next(color_cycle) for t in unique_tokens}
            
            # Group data by token
            token_groups = {t: {'x': [], 'y': [], 'z': []} for t in unique_tokens}
            is_grid_3d = (umap_dim == 3 and len(grid_tokens_filtered[0]) == 4)
            
            for item in grid_tokens_filtered:
                t = item[-1]
                token_groups[t]['x'].append(item[0])
                token_groups[t]['y'].append(item[1])
                if is_grid_3d:
                    token_groups[t]['z'].append(item[2])
            
            # Create one trace per token
            for t in unique_tokens:
                group = token_groups[t]
                color = token_to_color[t]
                
                if is_grid_3d:
                    grid_trace = go.Scatter3d(
                        x=group['x'], y=group['y'], z=group['z'],
                        mode='markers',
                        marker=dict(size=4, color=color, opacity=0.15, symbol='circle'), # Semi-transparent points
                        text=[t] * len(group['x']),
                        hoverinfo='text',
                        name=f'Region: {t}',
                        showlegend=True,
                        legendgroup=f'region_{t}'
                    )
                else:
                    grid_trace = go.Scatter(
                        x=group['x'], y=group['y'],
                        mode='markers',
                        marker=dict(size=8, color=color, opacity=0.15, symbol='square'), # Semi-transparent squares as regions
                        text=[t] * len(group['x']),
                        hoverinfo='text',
                        name=f'Region: {t}',
                        showlegend=True,
                        legendgroup=f'region_{t}'
                    )
                fig.add_trace(grid_trace)
    if anchor_embedding is not None and anchor_labels is not None:
        anchor_x = anchor_embedding[:, 0]
        anchor_y = anchor_embedding[:, 1]
        anchor_z = anchor_embedding[:, 2] if (umap_dim == 3 and anchor_embedding.shape[1] >= 3) else None
        anchor_texts = [str(int(lbl)) for lbl in anchor_labels]
        
        if umap_dim == 3 and anchor_embedding.shape[1] >= 3:
            anchor_trace = go.Scatter3d(
                x=anchor_x,
                y=anchor_y,
                z=anchor_z,
                mode='text',
                text=anchor_texts,
                textfont=dict(size=16, color='black', family='Arial Black'),
                hovertemplate='Token Anchor: %{text}<extra></extra>',
                hoverlabel=dict(bgcolor='black', font_color='white'),
                name='Token Anchors',
                showlegend=True
            )
        else:
            anchor_trace = go.Scatter(
                x=anchor_x,
                y=anchor_y,
                mode='text',
                text=anchor_texts,
                textfont=dict(size=16, color='black', family='Arial Black'),
                hovertemplate='Token Anchor: %{text}<extra></extra>',
                hoverlabel=dict(bgcolor='black', font_color='white'),
                name='Token Anchors',
                showlegend=True
            )
        fig.add_trace(anchor_trace)
    
    # Save HTML (enable scroll zoom and responsive layout)
    os.makedirs(save_dir, exist_ok=True)
    dim_suffix = "3D" if umap_dim == 3 else "2D"
    filename = f"umap{dim_suffix}_pos{position}_layer{layer_idx}_{marker_mode}{pred_filter_suffix}.html"
    save_path = os.path.join(save_dir, filename)
    
    # config enables scroll zoom; responsive makes the chart follow browser window size
    fig.write_html(
        save_path,
        config={
            'scrollZoom': True,  # Enable scroll zoom
            'responsive': True,  # Responsive layout
            'displayModeBar': True,  # Show toolbar
        },
        full_html=True,
        include_plotlyjs='cdn'  # Use CDN to reduce file size
    )
    print(f"Interactive HTML saved to: {save_path}")
    
    return save_path
