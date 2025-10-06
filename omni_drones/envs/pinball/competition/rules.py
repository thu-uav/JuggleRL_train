import torch


def in_half_court_1(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] > (-L / 2))
        & (pos[..., 0] < 0)
        & (pos[..., 1] > (-W / 2))
        & (pos[..., 1] < (W / 2))
    )


def in_half_court_0(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] > 0)
        & (pos[..., 0] < (L / 2))
        & (pos[..., 1] > (-W / 2))
        & (pos[..., 1] < (W / 2))
    )


def _not_in_bounds(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    """_summary_

    Args:
        pos (torch.Tensor): (*,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: (*,1)
    """
    boundary_xy = (
        torch.tensor([L / 2, W / 2],
                     device=pos.device).unsqueeze(0).unsqueeze(0)
    )  # (1,1,2)
    pos_xy = pos[..., :2].abs()  # (*,2)

    tmp = (pos_xy > boundary_xy).any(-1)  # (*,1)
    return tmp


def team_0_wins(
    ball_pos: torch.Tensor,
    drone_pos: torch.Tensor,
    team_turn: torch.Tensor,
    ball_hit_net: torch.Tensor,
    L: float,
    W: float,
    wrong_hit: torch.Tensor,
    num_hits: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        ball_pos (torch.Tensor): (E,1,3) float
        team_turn (torch.Tensor): (E,)

    Returns:
        torch.Tensor: (E,1)
    """

    ball_hit_ground = ball_pos[..., 2] < 0.3  # (E,1)

    # 球落在了team1的半场，无论是谁发的，team0都赢了
    case_1 = ball_hit_ground & in_half_court_1(ball_pos, L, W)
    # 或team1回球出界
    case_2 = _not_in_bounds(ball_pos, L, W) & (team_turn == 0).unsqueeze(-1)
    # 或team1把球打到了网上（不考虑弹过去。按照目前的初始化设定，发球失败，球是不可能自己碰到网的)
    case_3 = ball_hit_net * (team_turn == 0).unsqueeze(-1) 
    case_3 = torch.where(num_hits == 0, torch.zeros_like(case_3), case_3) # 无人机0发球，没击到球（或者击到球，但没检测到），此时不可能无人机0可以通过case_3获胜
    # team1连续击球两次
    case_4 = wrong_hit & (team_turn == 0).unsqueeze(-1)
    # team1无人机飞机位置太低
    case_5 = (drone_pos[:, 1, 2] < 0.2).unsqueeze(-1)
   

    win = case_1 | case_2 | case_3 | case_4 | case_5

    return win.long(), case_1.long(), case_2.long(), case_3.long(), case_4.long(), case_5.long()


def team_1_wins(
    ball_pos: torch.Tensor,
    drone_pos: torch.Tensor,
    team_turn: torch.Tensor,
    ball_hit_net: torch.Tensor,
    L: float,
    W: float,
    wrong_hit: torch.Tensor,
    num_hits: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        ball_pos (torch.Tensor): (E,1,3) float
        team_turn (torch.Tensor): (E,)

    Returns:
        torch.Tensor: (E,1)
    """

    ball_hit_ground = ball_pos[..., 2] < 0.3  # (E,1)

    # 球落在了team0的半场，无论是谁发的，team1都赢了
    case_1 = ball_hit_ground & in_half_court_0(ball_pos, L, W)
    # 或team0发球出界
    case_2 = _not_in_bounds(ball_pos, L, W) & (team_turn == 1).unsqueeze(-1)
    # 或team0把球打到了网上（不考虑弹过去。按照目前的初始化设定，发球失败，球是不可能自己碰到网的）
    case_3 = (ball_hit_net * (team_turn == 1).unsqueeze(-1)) | (
        ball_hit_net & (team_turn == 0).unsqueeze(-1) & (num_hits == 0) # 没检测到player0击球，但是player0发球上网
    )
    # team0连续击球两次
    case_4 = wrong_hit & (team_turn == 1).unsqueeze(-1)
    # team0无人机飞机位置太低
    case_5 = (drone_pos[:, 0, 2] < 0.2).unsqueeze(-1)

    win = case_1 | case_2 | case_3 | case_4 | case_5

    return win.long(), case_1.long(), case_2.long(), case_3.long(), case_4.long(), case_5.long()


def determine_game_result(
    ball_pos: torch.Tensor,
    drone_pos: torch.Tensor,
    turn: torch.Tensor,
    ball_hit_net: torch.Tensor,
    L: float,
    W: float,
    wrong_hit,
    num_hits,
) -> torch.Tensor:
    """_summary_

    Args:
        ball_pos (torch.Tensor): (E,1,3)
        turn (torch.Tensor):(E,)
        ball_hit_net (torch.Tensor): (E,1)

    Returns:
        torch.Tensor: (E,1) 1: drone0 wins 0: not over -1: drone1 wins
    """
    team_0_win, team_0_case_1, team_0_case_2, team_0_case_3, team_0_case_4, team_0_case_5 = team_0_wins(
        ball_pos=ball_pos, drone_pos=drone_pos, team_turn=turn, ball_hit_net=ball_hit_net, L=L, W=W, wrong_hit=wrong_hit, num_hits=num_hits,
    )
    team_1_win, team_1_case_1, team_1_case_2, team_1_case_3, team_1_case_4, team_1_case_5 = team_1_wins(
        ball_pos=ball_pos, drone_pos=drone_pos, team_turn=turn, ball_hit_net=ball_hit_net, L=L, W=W, wrong_hit=wrong_hit, num_hits=num_hits,
    )

    result = team_0_win - team_1_win

    return result, team_0_case_1, team_0_case_2, team_0_case_3, team_0_case_4, team_0_case_5, team_1_case_1, team_1_case_2, team_1_case_3, team_1_case_4, team_1_case_5


# 不然接球一方只要G的足够快就不会吃到失败惩罚
def misbehave_as_lose(drone_misbehave: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_misbehave (torch.Tensor): (E,2)

    Returns:
        torch.Tensor: (E,1)
    """
    tmp = - drone_misbehave[:, 0].long() + drone_misbehave[:, 1].long()  # (E,)

    return tmp.unsqueeze(-1)


def game_result_to_matrix(game_result: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]], device=game_result.device
    )
    return table[game_result.squeeze(-1) + 1]
