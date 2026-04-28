# ============================================================
# main_illustration_parts_R.R
# ============================================================
library(ggplot2)
library(grid)

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir <- "RPlots/Figure1_parts"
dir.create(out_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
gauss <- function(x, mu, sigma, A = 1) {
  A * exp(-0.5 * ((x - mu) / sigma)^2)
}

bandwave <- function(u, f = 1, A = 1, phase = 0, offset = 0) {
  offset + A * cos(2 * pi * f * u + phase)
}

theme_none <- function() {
  theme_void() +
    theme(
      plot.background  = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.position = "none",
      plot.margin = margin(0, 0, 0, 0)
    )
}

save_transparent <- function(filename, plot, width = 5, height = 5) {
  ggsave(
    file.path(out_dir, filename),
    plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "transparent"
  )
}

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
COL_NODE <- "#B3B3B3"
COL_EDGE <- "#ADADAD"
COL_ORIG <- "#667D9C"
COL_MOD  <- "#B8804D"

COL_LIGHT <- "#EAEAEA"
COL_MID   <- "#D1D1D1"
COL_DARK  <- "#B3B3B3"

# ------------------------------------------------------------
# Network data
# ------------------------------------------------------------
pts <- data.frame(
  id = 1:15,
  x = c(
    0.50, 0.72, 0.86, 0.87, 0.73, 0.28, 0.23, 0.10,
    0.50, 0.66, 0.60, 0.41, 0.33,
    0.47, 0.55
  ),
  y = c(
    0.91, 0.83, 0.66, 0.43, 0.22, 0.78, 0.23, 0.47,
    0.68, 0.58, 0.39, 0.34, 0.56,
    0.49, 0.51
  )
)

mod_node <- 11

edges <- data.frame(
  i = c(
    1, 2, 6, 7, 2, 5,
    9, 10, 12, 13, 10,
    14, 14, 15, 14,
    1, 2, 4, 5, 7, 8,
    11, 11, 11, 11
  ),
  j = c(
    2, 3, 7, 8, 4, 7,
    10, 11, 13, 9, 13,
    15, 9, 10, 12,
    9, 10, 10, 11, 12, 13,
    10, 12, 5, 15
  )
)

edges$modified <- edges$i == mod_node | edges$j == mod_node

edge_df <- merge(edges, pts, by.x = "i", by.y = "id")
edge_df <- merge(edge_df, pts, by.x = "j", by.y = "id", suffixes = c("1", "2"))

# ------------------------------------------------------------
# Panel 1: network only
# ------------------------------------------------------------
p_network <- ggplot() +
  geom_segment(
    data = subset(edge_df, !modified),
    aes(x = x1, y = y1, xend = x2, yend = y2),
    linewidth = 1.0,
    color = COL_EDGE,
    lineend = "round"
  ) +
  geom_segment(
    data = subset(edge_df, modified),
    aes(x = x1, y = y1, xend = x2, yend = y2),
    linewidth = 1.8,
    color = COL_MOD,
    lineend = "round"
  ) +
  geom_point(
    data = pts,
    aes(x = x, y = y),
    size = 5.5,
    color = COL_NODE
  ) +
  geom_point(
    data = subset(pts, id == mod_node),
    aes(x = x, y = y),
    size = 6.4,
    color = COL_MOD
  ) +
  coord_equal(xlim = c(-0.05, 1.15), ylim = c(-0.02, 1.02), expand = FALSE) +
  theme_none()

save_transparent("01_network.png", p_network, width = 5.2, height = 4.6)

# ------------------------------------------------------------
# Panel 2: matrix only
# ------------------------------------------------------------
n <- nrow(pts)
M <- matrix(0, n, n)

diag(M) <- 0.55

base_edges <- subset(edges, !modified)
mod_edges  <- subset(edges, modified)

for (k in seq_len(nrow(base_edges))) {
  i <- base_edges$i[k]
  j <- base_edges$j[k]
  M[i, j] <- 0.72
  M[j, i] <- 0.72
}

for (k in seq_len(nrow(mod_edges))) {
  i <- mod_edges$i[k]
  j <- mod_edges$j[k]
  M[i, j] <- 0.98
  M[j, i] <- 0.98
}

mat_df <- expand.grid(i = 1:n, j = 1:n)
mat_df$value <- as.vector(M)
mat_df$modified <- mat_df$i == mod_node | mat_df$j == mod_node

mat_df$fill <- ifelse(
  mat_df$value == 0, COL_LIGHT,
  ifelse(
    mat_df$modified, COL_MOD,
    ifelse(mat_df$value > 0.78, COL_DARK,
           ifelse(mat_df$value > 0.60, COL_MID, COL_LIGHT))
  )
)

highlight_df <- rbind(
  data.frame(
    xmin = 0.5,
    xmax = n + 0.5,
    ymin = mod_node - 0.5,
    ymax = mod_node + 0.5
  ),
  data.frame(
    xmin = mod_node - 0.5,
    xmax = mod_node + 0.5,
    ymin = 0.5,
    ymax = n + 0.5
  )
)

p_matrix <- ggplot() +
  geom_tile(
    data = mat_df,
    aes(x = j, y = i),
    fill = COL_LIGHT,
    color = "transparent",
    width = 1,
    height = 1
  ) +
  geom_rect(
    data = highlight_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = COL_MOD,
    alpha = 0.14,
    color = COL_MOD,
    linewidth = 0.45
  ) +
  geom_tile(
    data = subset(mat_df, value > 0),
    aes(x = j, y = i, fill = fill),
    color = "transparent",
    width = 0.78,
    height = 0.78
  ) +
  scale_fill_identity() +
  coord_equal(
    xlim = c(0.3, n + 0.7),
    ylim = c(n + 0.7, 0.3),
    expand = FALSE
  ) +
  theme_none()

save_transparent("02_matrix.png", p_matrix, width = 5.2, height = 5.2)

# ------------------------------------------------------------
# Panel 3: time response only
# ------------------------------------------------------------
t <- seq(0, 10, length.out = 900)

y0 <- 0.75
y_orig <- y0 * exp(-t / 2.15)
y_mod <- y0 * exp(-t / 3.2) +
  1.05 * (1 - exp(-t / 0.9)) * exp(-t / 1.5)

time_df <- rbind(
  data.frame(t = t, y = y_orig, group = "orig"),
  data.frame(t = t, y = y_mod,  group = "mod")
)

p_time <- ggplot(time_df, aes(t, y, color = group)) +
  geom_line(linewidth = 1.4) +
  scale_color_manual(values = c(orig = COL_ORIG, mod = COL_MOD)) +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 1.18), expand = FALSE) +
  theme_none() +
  theme(
    axis.line.x = element_line(color = "black", linewidth = 0.6),
    axis.line.y = element_line(color = "black", linewidth = 0.6)
  )

save_transparent("03_time_response.png", p_time, width = 5.8, height = 4.2)

# ------------------------------------------------------------
# Panel 4: modal contribution inset only
# ------------------------------------------------------------
orange_frac <- data.frame(
  mode = c(1, 2, 4),
  orange = c(0.10, 0.40, 0.18)
)

bars <- do.call(
  rbind,
  lapply(seq_len(nrow(orange_frac)), function(k) {
    mode <- orange_frac$mode[k]
    h_orange <- orange_frac$orange[k]
    h_gray_total <- 1 - h_orange
    h_gray_bottom <- 0.55 * h_gray_total
    h_gray_top <- 0.45 * h_gray_total
    
    data.frame(
      mode = mode,
      ymin = c(0, h_gray_bottom, h_gray_bottom + h_orange),
      ymax = c(h_gray_bottom, h_gray_bottom + h_orange, 1),
      fill = c("#B8B8B8", COL_MOD, "#B8B8B8")
    )
  })
)

p_modes <- ggplot(bars) +
  geom_rect(
    aes(
      xmin = mode - 0.29,
      xmax = mode + 0.29,
      ymin = ymin,
      ymax = ymax,
      fill = fill
    ),
    color = NA
  ) +
  geom_rect(
    data = data.frame(mode = c(1, 2, 4)),
    aes(
      xmin = mode - 0.29,
      xmax = mode + 0.29,
      ymin = 0,
      ymax = 1
    ),
    fill = NA,
    color = "#333333",
    linewidth = 0.35
  ) +
  geom_point(
    data = data.frame(x = c(2.88, 3.00, 3.12), y = c(0.36, 0.50, 0.64)),
    aes(x, y),
    size = 1.6,
    color = "#595959"
  ) +
  scale_fill_identity() +
  coord_cartesian(xlim = c(0.35, 4.65), ylim = c(0, 1.34), expand = FALSE) +
  theme_none()

save_transparent("04_modal_contributions.png", p_modes, width = 3.8, height = 2.8)

# ------------------------------------------------------------
# Panel 5: timescale decomposition only
# no color bands, no text
# ------------------------------------------------------------
# ------------------------------------------------------------
# Panel 5: separate timescale decomposition plots
# one file each: slow, mid, fast
# ------------------------------------------------------------
make_wave_panel <- function(y_orig, y_mod, filename) {
  df <- rbind(
    data.frame(u = u, y = y_orig, group = "orig"),
    data.frame(u = u, y = y_mod,  group = "mod")
  )
  
  p <- ggplot(df, aes(u, y, color = group)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = c(orig = COL_ORIG, mod = COL_MOD)) +
    coord_cartesian(xlim = c(0, 1), ylim = c(-0.07, 0.07), expand = FALSE) +
    theme_none()
  
  save_transparent(filename, p, width = 3.8, height = 1.2)
}

make_wave_panel(slow_orig, slow_mod, "05a_slow_wave.png")
make_wave_panel(mid_orig,  mid_mod,  "05b_mid_wave.png")
make_wave_panel(fast_orig, fast_mod, "05c_fast_wave.png")

save_transparent("05_timescale_decomposition.png", p_bridge, width = 3.8, height = 4.2)

# ------------------------------------------------------------
# Panel 6: frequency profile only
# no color bands, no text
# ------------------------------------------------------------
omega <- exp(seq(log(1e-2), log(1e2), length.out = 900))
x <- log10(omega)

S_orig <- 0.40 +
  gauss(x, -0.95, 0.25, 0.18) +
  gauss(x, -0.15, 0.28, 0.46) -
  0.10 / (1 + exp(-4.0 * (x - 0.45)))

S_mod <- 0.19 +
  0.16 / (1 + (omega / 0.10)^0.85) +
  0.035 * exp(-(omega / 1.8)^0.7)

freq_df <- rbind(
  data.frame(omega = omega, y = S_orig, group = "orig"),
  data.frame(omega = omega, y = S_mod,  group = "mod")
)

p_freq <- ggplot(freq_df, aes(omega, y, color = group)) +
  geom_line(linewidth = 1.4) +
  scale_x_log10() +
  scale_color_manual(values = c(orig = COL_ORIG, mod = COL_MOD)) +
  coord_cartesian(xlim = c(min(omega), max(omega)), ylim = c(0, 1.05), expand = FALSE) +
  theme_none() +
  theme(
    axis.line.x = element_line(color = "black", linewidth = 0.6),
    axis.line.y = element_line(color = "black", linewidth = 0.6)
  )

save_transparent("06_frequency_profile.png", p_freq, width = 5.8, height = 4.2)
