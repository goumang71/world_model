"""Display-only dashboard; never modifies pixels passed into the policy."""
from pathlib import Path


def driving_metrics(env):
    base = env.unwrapped
    road_wheels = sum(bool(wheel.tiles) for wheel in base.car.wheels)
    return dict(track_coverage=float(base.tile_visited_count / len(base.track)),
                road_wheels=road_wheels, off_track=road_wheels < 4,
                fully_off_track=road_wheels == 0)


class DrivingHUD:
    def __init__(self, human=False):
        import pygame
        self.pg = pygame
        pygame.font.init()
        font = next((str(p) for p in [Path('C:/Windows/Fonts/msyh.ttc'),
                                     Path('C:/Windows/Fonts/simhei.ttf')] if p.exists()), None)
        self.chinese = font is not None
        self.font = pygame.font.Font(font, 19 if font else 24)
        self.human = human
        self.screen = None
        self.clock = pygame.time.Clock()

    def frame(self, rgb, *, step, seed, total, metrics, lap_finished=False,
              terminated=False, truncated=False):
        pg = self.pg
        height = 108
        surface = pg.Surface((rgb.shape[1], rgb.shape[0] + height))
        surface.fill((20, 25, 34))
        surface.blit(pg.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, height))
        wheels = metrics['road_wheels']
        if self.chinese:
            road = '在赛道内' if wheels == 4 else ('完全离开赛道' if wheels == 0 else '部分车轮离开赛道')
            state = '已完圈' if lap_finished else ('环境终止' if terminated else ('达到步数上限' if truncated else '行驶中'))
            lines = [f'种子 {seed}    步数 {step}/1000    累计得分 {total:.1f}',
                     f'赛道覆盖率 {metrics["track_coverage"]:.1%}    状态：{state}',
                     f'{road}    接触赛道的车轮：{wheels}/4']
        else:
            road = 'ON TRACK' if wheels == 4 else ('OFF TRACK' if wheels == 0 else 'PARTLY OFF TRACK')
            state = 'LAP FINISHED' if lap_finished else ('TERMINATED' if terminated else ('STEP LIMIT' if truncated else 'RUNNING'))
            lines = [f'Seed {seed}   Steps {step}/1000   Return {total:.1f}',
                     f'Track coverage {metrics["track_coverage"]:.1%}   {state}',
                     f'{road}   Wheels on road: {wheels}/4']
        for i, line in enumerate(lines):
            color = ((255, 125, 100) if wheels < 4 else (125, 230, 170)) if i == 2 else (235, 240, 248)
            surface.blit(self.font.render(line, True, color), (12, 5 + i * 33))
        closed = False
        if self.human:
            if self.screen is None:
                pg.display.init()
                self.screen = pg.display.set_mode(surface.get_size())
                pg.display.set_caption('World Models - Driving Monitor')
            closed = any(event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE)
                         for event in pg.event.get())
            if not closed:
                self.screen.blit(surface, (0, 0))
                pg.display.flip()
                self.clock.tick(50)
        return pg.surfarray.array3d(surface).swapaxes(0, 1), closed

    def close(self):
        if self.human:
            self.pg.display.quit()
