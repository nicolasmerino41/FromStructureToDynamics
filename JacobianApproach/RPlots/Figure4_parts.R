# ============================================================
# Separate panels: uni plots, multi, profile, community response
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------
# Parameters
# -------------------------

omega_low  <- 0.08
omega_mid  <- 0.62
omega_high <- 2.8

d_low  <- 5.0
d_mid  <- 5e-5
d_high <- 5.0

eps12 <- 1e-8
eps23 <- 1e-8

dt <- 0.02
tmax <- 600
burn_time <- 300

low_col  <- "#2C7FB8"
mid_col  <- "#D95F02"
high_col <- "#1B9E77"
resp_col <- "#7B3294"

# -------------------------
# Matrix A
# -------------------------

A <- matrix(c(
  -d_low, 0, eps12, 0, 0, 0,
  0, -d_low, 0, eps12, 0, 0,
  eps12, 0, -d_mid, -omega_mid, eps23, 0,
  0, eps12, omega_mid, -d_mid, 0, eps23,
  0, 0, eps23, 0, -d_high, 0,
  0, 0, 0, eps23, 0, -d_high
), nrow = 6, byrow = TRUE)

# -------------------------
# Resolvent profile
# -------------------------

resolvent_norm <- function(A, omega) {
  n <- nrow(A)
  M <- 1i * omega * diag(n) - A
  R <- solve(M)
  svd(R, nu = 0, nv = 0)$d[1]
}

omega_s <- exp(seq(log(0.03), log(8), length.out = 12000))
S <- vapply(omega_s, function(w) resolvent_norm(A, w), numeric(1))

df_profile <- tibble(
  omega = omega_s,
  S = S
)

profile_low_line  <- 0.09
profile_mid_line  <- 0.62
profile_high_line <- 3.8

# -------------------------
# Forcing
# -------------------------

b <- c(0.35, -0.25, 1.00, -0.85, 0.30, -0.20)
b <- b / sqrt(sum(b^2))

a_low  <- 0.12
a_mid  <- 0.12
a_high <- 0.12

phi_low  <- 0.4
phi_mid  <- 1.1
phi_high <- -0.7

low_component <- function(t) {
  a_low * sin(omega_low * t + phi_low)
}

mid_component <- function(t) {
  a_mid * sin(omega_mid * t + phi_mid)
}

high_component <- function(t) {
  a_high * sin(omega_high * t + phi_high)
}

forcing_scalar <- function(t) {
  low_component(t) + mid_component(t) + high_component(t)
}

forcing_fun <- function(t) {
  forcing_scalar(t) * b
}

# -------------------------
# RK4 simulation
# -------------------------

simulate_system <- function(A, forcing_fun, dt, tmax) {
  ts <- seq(0, tmax, by = dt)
  n <- nrow(A)
  X <- matrix(0, nrow = length(ts), ncol = n)
  
  f <- function(x, t) {
    as.numeric(A %*% x + forcing_fun(t))
  }
  
  x <- rep(0, n)
  
  for (k in seq_len(length(ts) - 1)) {
    t <- ts[k]
    
    k1 <- f(x, t)
    k2 <- f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 <- f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 <- f(x + dt * k3, t + dt)
    
    x <- x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X[k + 1, ] <- x
  }
  
  list(time = ts, X = X)
}

sim <- simulate_system(A, forcing_fun, dt, tmax)

ts <- sim$time
X <- sim$X

c_readout <- c(0, 0, 1, 1, 0, 0)
response <- as.numeric(X %*% c_readout)

df_time <- tibble(
  time = ts,
  slow = low_component(ts),
  intermediate = mid_component(ts),
  fast = high_component(ts),
  forcing = forcing_scalar(ts),
  response = response
) |>
  filter(time >= burn_time)

force_scale <- max(abs(df_time$forcing), na.rm = TRUE)
resp_scale  <- max(abs(df_time$response), na.rm = TRUE)

df_time <- df_time |>
  mutate(response_scaled = response / resp_scale * force_scale)

# -------------------------
# Theme
# -------------------------

theme_panel <- theme_void(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(8, 8, 8, 8)
  )

# ============================================================
# Separate plots
# ============================================================

p_uni_slow <- ggplot(df_time, aes(time, slow)) +
  geom_line(color = low_col, linewidth = 0.9) +
  coord_cartesian(xlim = c(300, 365)) +
  theme_panel

p_uni_intermediate <- ggplot(df_time, aes(time, intermediate)) +
  geom_line(color = mid_col, linewidth = 0.9) +
  coord_cartesian(xlim = c(300, 365)) +
  theme_panel

p_uni_fast <- ggplot(df_time, aes(time, fast)) +
  geom_line(color = high_col, linewidth = 0.9) +
  coord_cartesian(xlim = c(300, 365)) +
  theme_panel

p_multi <- ggplot(df_time, aes(time, forcing)) +
  geom_line(color = "black", linewidth = 1.05) +
  coord_cartesian(
    xlim = c(300, 365),
    ylim = c(-1.25 * force_scale, 1.25 * force_scale)
  ) +
  theme_panel

p_profile <- ggplot(df_profile, aes(omega, S)) +
  geom_line(linewidth = 0.9, color = "black") +
  geom_vline(xintercept = profile_low_line,
             linetype = "dashed", color = low_col, linewidth = 1.05) +
  geom_vline(xintercept = profile_mid_line,
             linetype = "dashed", color = mid_col, linewidth = 1.05) +
  geom_vline(xintercept = profile_high_line,
             linetype = "dashed", color = high_col, linewidth = 1.05) +
  scale_x_log10(limits = c(0.03, 8), breaks = NULL) +
  scale_y_log10() +
  labs(x = expression(omega), y = "sensitivity") +
  theme_classic(base_size = 11) +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_line(linewidth = 0.35),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(4, 4, 4, 4)
  )

# -------------------------
# Delayed perturbation for community plot only
# -------------------------
# -------------------------
# Community plot with delayed response onset
# -------------------------
perturbation_start <- 309.5
ramp_tau <- 6   # larger = slower ramp-up

df_community <- df_time |>
  mutate(
    time_since_perturbation = pmax(0, time - perturbation_start),
    
    ramp = ifelse(
      time < perturbation_start,
      0,
      1 - exp(-time_since_perturbation / ramp_tau)
    ),
    
    perturbation_delayed = ifelse(
      time < perturbation_start,
      NA,
      forcing
    ),
    
    response_delayed = ramp * force_scale * 0.85 *
      sin(omega_mid * time_since_perturbation + phi_mid)
  )

p_community <- ggplot(df_community, aes(time)) +
  geom_line(
    aes(y = perturbation_delayed),
    color = "grey72",
    linewidth = 0.55,
    alpha = 0.65,
    na.rm = TRUE
  ) +
  geom_line(
    aes(y = response_delayed),
    color = resp_col,
    linewidth = 1.35
  ) +
  coord_cartesian(
    xlim = c(300, 365),
    ylim = c(-1.35 * force_scale, 1.35 * force_scale)
  ) +
  theme_panel +
  theme(legend.position = "none")
# ============================================================
# Separate network plot
# ============================================================

COL_EDGE <- "#ABABAB"
COL_NODE <- "#2E2E2E"

pts <- data.frame(
  node = 1:12,
  x = c(0.08, 0.18, 0.29, 0.22,
        0.42, 0.50, 0.46,
        0.70, 0.83, 0.73,
        1.03, 1.00),
  y = c(0.63, 0.82, 0.60, 0.34,
        0.80, 0.54, 0.22,
        0.74, 0.54, 0.28,
        0.73, 0.40)
)

x_center <- mean(range(pts$x))

pts_net <- transform(
  pts,
  x = 0.5 + 0.78 * (x - x_center),
  y = y - 0.08
)

base_edges <- data.frame(
  from = c(1,2,2,3,4,5,5,6,6,6,8,9,9,9,9),
  to   = c(2,3,5,6,6,6,9,7,9,10,9,10,11,12,8)
)

edge_df <- function(edges, pts_df) {
  merge(edges, pts_df, by.x = "from", by.y = "node") |>
    merge(pts_df, by.x = "to", by.y = "node", suffixes = c("_from", "_to"))
}

base_edges_df <- edge_df(base_edges, pts_net)

p_network <- ggplot() +
  geom_segment(
    data = base_edges_df,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = "black",
    linewidth = 1.0
  ) +
  geom_point(
    data = pts_net,
    aes(x, y),
    color = COL_NODE,
    size = 5
  ) +
  coord_cartesian(xlim = c(0.02, 1.04), ylim = c(0.02, 0.86), expand = FALSE) +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(4, 4, 4, 4)
  )

p_network

ggsave(
  "RPlots/separate_panels/network.png",
  p_network,
  width = 3.2,
  height = 2.4,
  dpi = 450
)
# ============================================================
# Display separately
# ============================================================

p_uni_slow
p_uni_intermediate
p_uni_fast
p_multi
p_profile
p_community

# ============================================================
# Save separately
# ============================================================

dir.create("RPlots/separate_panels", recursive = TRUE, showWarnings = FALSE)

ggsave("RPlots/separate_panels/uni_slow.png", p_uni_slow,
       width = 4.2, height = 1.2, dpi = 450)

ggsave("RPlots/separate_panels/uni_intermediate.png", p_uni_intermediate,
       width = 4.2, height = 1.2, dpi = 450)

ggsave("RPlots/separate_panels/uni_fast.png", p_uni_fast,
       width = 4.2, height = 1.2, dpi = 450)

ggsave("RPlots/separate_panels/multifrequency.png", p_multi,
       width = 4.8, height = 2.0, dpi = 450)

ggsave("RPlots/separate_panels/profile.png", p_profile,
       width = 7.0, height = 3.0, dpi = 450)

ggsave("RPlots/separate_panels/community_response.png", p_community,
       width = 4.8, height = 2.0, dpi = 450)
