import asyncio
import time
import traceback
import types
import random
import builtins as _py_builtins
import hashlib
import marshal
from typing import List, Tuple, Set, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- DB (SQLAlchemy) ---
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, select
from sqlalchemy.orm import sessionmaker

# --- Security (Password) ---
from passlib.context import CryptContext

# --- Multiprocessing (격리 실행) ---
import multiprocessing as mp


# =========================
# 1) 최소 위험 + 대부분 허용 샌드박스
# =========================


ALLOWED_MODULES = {
    "math", "random", "statistics", "itertools", "functools",
    "collections", "heapq", "bisect",
}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed")
    return _py_builtins.__import__(name, globals, locals, fromlist, level)

def _build_allowed_builtins():
    allowed = {
        "abs": _py_builtins.abs, "all": _py_builtins.all, "any": _py_builtins.any,
        "bin": _py_builtins.bin, "bool": _py_builtins.bool, "bytearray": _py_builtins.bytearray,
        "bytes": _py_builtins.bytes, "callable": _py_builtins.callable, "chr": _py_builtins.chr,
        "complex": _py_builtins.complex, "divmod": _py_builtins.divmod, "enumerate": _py_builtins.enumerate,
        "float": _py_builtins.float, "format": _py_builtins.format, "frozenset": _py_builtins.frozenset,
        "hash": _py_builtins.hash, "hex": _py_builtins.hex, "int": _py_builtins.int,
        "isinstance": _py_builtins.isinstance, "issubclass": _py_builtins.issubclass,
        "iter": _py_builtins.iter, "len": _py_builtins.len, "list": _py_builtins.list,
        "map": _py_builtins.map, "max": _py_builtins.max, "min": _py_builtins.min,
        "next": _py_builtins.next, "object": _py_builtins.object, "ord": _py_builtins.ord,
        "pow": _py_builtins.pow, "print": _py_builtins.print, "range": _py_builtins.range,
        "repr": _py_builtins.repr, "reversed": _py_builtins.reversed, "round": _py_builtins.round,
        "set": _py_builtins.set, "slice": _py_builtins.slice, "sorted": _py_builtins.sorted,
        "str": _py_builtins.str, "sum": _py_builtins.sum, "tuple": _py_builtins.tuple,
        "type": _py_builtins.type, "zip": _py_builtins.zip,
        "__import__": _guarded_import,
    }
    for name in (
        "open", "eval", "exec", "compile", "input",
        "help", "dir", "globals", "locals", "vars",
        "breakpoint", "__loader__", "__spec__", "__build_class__",
    ):
        allowed.pop(name, None)
    return allowed

def load_strategy_from_string(code_string: str):
    """부모 프로세스 내에서 안전 빌트인으로 컴파일/exec하여 함수 객체만 가져온다."""
    try:
        sandbox_globals = {"__builtins__": _build_allowed_builtins()}
        local_scope = {}
        byte_code = compile(code_string, "<user_code>", "exec")
        exec(byte_code, sandbox_globals, local_scope)
        strategy_func = local_scope.get("strategy")
        if isinstance(strategy_func, types.FunctionType):
            return strategy_func, None
        return None, "Python 코드에서 'strategy(opponent_history, my_history)' 함수를 찾을 수 없습니다."
    except Exception as e:
        return None, f"코드 컴파일/실행 오류: {e}\n{traceback.format_exc()}"


# =========================
# 2) 기본 설정 및 상수
# =========================
#DATABASE_URL = "sqlite+aiosqlite:///./tournament.db"
def to_asyncpg_url(url: str) -> str:
    """Render가 주는 postgres 스킴을 async 드라이버용으로 치환."""
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url  # 이미 postgresql+asyncpg:// 인 경우 등

DATABASE_URL = to_asyncpg_url(os.environ.get("DATABASE_URL"))

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

PAYOFFS = {
    ('협력', '협력'): (3, 3),
    ('협력', '배신'): (0, 5),
    ('배신', '협력'): (5, 0),
    ('배신', '배신'): (1, 1)
}

MATCH_TIME_BUDGET_SEC = 10
PER_CALL_HARD_CAP_SEC = 5
RUNNER_STARTUP_GRACE_SEC = 8.0  # 데몬 러너 최초 시작만 적용
SANITY_CHECK_TIMEOUT_SEC = 5.0

# ns 단위
MATCH_TIME_BUDGET_NS = int(MATCH_TIME_BUDGET_SEC * 1e9)
PER_CALL_HARD_CAP_NS = int(PER_CALL_HARD_CAP_SEC * 1e9)

tournament_lock = asyncio.Lock()

# ⚠️ CORS (개발 편의용 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 3) 데이터베이스 모델 (테이블)
# =========================
class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    code_string = Column(String, nullable=False)
    total_score = Column(Integer, default=0, nullable=False)
    avg_score = Column(Integer, default=0, nullable=False)
    error_flag = Column(Boolean, default=False, nullable=False)
    error_message = Column(String, nullable=True)


# =========================
# 4) Pydantic 모델 (API 입/출력)
# =========================
class Submission(BaseModel):
    user_name: str
    password: str
    code: str

class ScoreboardEntry(BaseModel):
    user_name: str
    avg_score: int
    class Config:
        orm_mode = True


# =========================
# 5) 스타트업 (DB 생성 + mp start method 설정)
# =========================
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.on_event("startup")
async def startup():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.tournament_running = False
    app.state.tournament_started_at = None
    app.state.tournament_last_finished_at = None
    app.state.tournament_task = asyncio.create_task(tournament_scheduler())

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)


# =========================
# 6) 전략별 상시 데몬 러너 (Pipe + ready + marshal 바이트코드)
# =========================
def _runner_proc(code_bytes: bytes, conn):
    try:
        sandbox_globals = {"__builtins__": _build_allowed_builtins()}
        local_scope = {}
        code_obj = marshal.loads(code_bytes)
        exec(code_obj, sandbox_globals, local_scope)

        strategy_func = local_scope.get("strategy")
        if not isinstance(strategy_func, types.FunctionType):
            conn.send(("error", "Python 코드에서 'strategy(opponent_history, my_history)' 함수를 찾을 수 없습니다."))
            return

        conn.send(("ready", None))  # 로드 완료

        while True:
            msg = conn.recv()
            if msg == ("cmd", "quit"):
                break
            tag, opp_hist, my_hist = msg
            try:
                move = strategy_func(tuple(opp_hist), tuple(my_hist))
                conn.send(("ok", move))
            except Exception as e:
                conn.send(("error", f"{e}\n{traceback.format_exc()}"))
    except Exception as e:
        try:
            conn.send(("error", f"{e}\n{traceback.format_exc()}"))
        except:
            pass
    finally:
        try:
            conn.close()
        except:
            pass

class PlayerRunner:
    """전략별 상시 데몬 러너. 앱 생존 동안 재사용."""
    def __init__(self, code_string: str, startup_grace_sec: float = RUNNER_STARTUP_GRACE_SEC):
        self.ctx = mp.get_context("spawn")
        self.parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        byte_code = compile(code_string, "<user_code>", "exec")
        code_bytes = marshal.dumps(byte_code)
        self.p = self.ctx.Process(target=_runner_proc, args=(code_bytes, child_conn))
        self.p.start()

        # ready 대기
        if not self.parent_conn.poll(startup_grace_sec):
            self.terminate()
            raise TimeoutError("strategy process startup exceeded grace time")
        status, payload = self.parent_conn.recv()
        if status == "error":
            self.terminate()
            raise RuntimeError(payload)

    def move(self, opp_hist, my_hist, timeout_sec: float):
        if not self.p.is_alive():
            raise RuntimeError("strategy process is not alive")
        self.parent_conn.send(("move", opp_hist, my_hist))
        if not self.parent_conn.poll(timeout_sec):
            self.terminate()
            raise TimeoutError(f"move exceeded {timeout_sec:.6f}s (killed process)")
        status, payload = self.parent_conn.recv()
        if status == "ok":
            return payload
        else:
            self.terminate()
            raise RuntimeError(payload)

    def terminate(self):
        if self.p.is_alive():
            try:
                self.parent_conn.send(("cmd", "quit"))
            except Exception:
                pass
            self.p.join(timeout=0.05)
            if self.p.is_alive():
                self.p.terminate()
                self.p.join()
        try:
            self.parent_conn.close()
        except:
            pass


# --- 글로벌 러너/해시/샌티티 캐시 ---
_runner_registry: Dict[int, PlayerRunner] = {}
_runner_codehash: Dict[int, str] = {}

# SANITY 캐시: (strategy_id, codehash) -> (ok: bool, message: Optional[str])
_sanity_cache: Dict[Tuple[int, str], Tuple[bool, Optional[str]]] = {}

def _code_hash(code_string: str) -> str:
    return hashlib.sha256(code_string.encode("utf-8")).hexdigest()

def get_or_create_runner(strategy_id: int, code_string: str) -> PlayerRunner:
    """같은 전략ID에 코드가 같으면 재사용, 바뀌면 교체."""
    h = _code_hash(code_string)
    runner = _runner_registry.get(strategy_id)
    ch = _runner_codehash.get(strategy_id)
    if runner is not None and ch == h:
        if not runner.p.is_alive():
            try:
                runner.terminate()
            except:
                pass
            runner = PlayerRunner(code_string)
            _runner_registry[strategy_id] = runner
        return runner
    if runner is not None:
        try:
            runner.terminate()
        except:
            pass
    runner = PlayerRunner(code_string)
    _runner_registry[strategy_id] = runner
    _runner_codehash[strategy_id] = h
    return runner

def terminate_all_runners():
    for r in list(_runner_registry.values()):
        try:
            r.terminate()
        except:
            pass
    _runner_registry.clear()
    _runner_codehash.clear()

# SANITY 체크(제출 시에만 사용)
def run_sanity_check(code_string: str) -> Tuple[bool, Optional[str]]:
    """간단 컴파일 + 1회 호출로 빠르게 검증."""
    try:
        func, err = load_strategy_from_string(code_string)
        if err:
            return False, f"[Compile Error] {err}"
        # 간단 호출 (무한루프 같은 건 여기서 잘 안 잡힐 수 있지만, 제출 시 최소 필터)
        _ = func(tuple(), tuple())
        return True, None
    except Exception as e:
        return False, f"[Sanity Check Error] {e}\n{traceback.format_exc()}"


# =========================
# 7) 게임 & 토너먼트 (SANITY는 재사용, eager warmup, 데몬 러너)
# =========================
class StrategyTimeoutError(Exception):
    pass

# 고속 풀리그(프로세스 없이 직접 실행)
def play_match_worker_no_signal(args: Tuple) -> Tuple:
    s1_id, s1_code, s1_name, s2_id, s2_code, s2_name = args
    func1, _ = load_strategy_from_string(s1_code)
    func2, _ = load_strategy_from_string(s2_code)

    history1, history2 = [], []
    score1, score2 = 0, 0
    matchnum = random.randint(80, 120)
    remaining1 = MATCH_TIME_BUDGET_SEC
    remaining2 = MATCH_TIME_BUDGET_SEC
    err_func, err_msg = None, None

    for _ in range(matchnum):
        # P1
        try:
            if remaining1 <= 0.0:
                raise StrategyTimeoutError(f"총 시간 예산 초과 ({MATCH_TIME_BUDGET_SEC}초)")
            t0 = time.perf_counter()
            move1 = func1(tuple(history2), tuple(history1))
            remaining1 -= (time.perf_counter() - t0)
            if move1 not in ['협력', '배신']:
                raise ValueError("반환값은 '협력' 또는 '배신'이어야 합니다.")
        except Exception as e:
            err_func, err_msg = 'func1', f"[vs {s2_name}] {e}"
            break
        # P2
        try:
            if remaining2 <= 0.0:
                raise StrategyTimeoutError(f"총 시간 예산 초과 ({MATCH_TIME_BUDGET_SEC}초)")
            t0 = time.perf_counter()
            move2 = func2(tuple(history1), tuple(history2))
            remaining2 -= (time.perf_counter() - t0)
            if move2 not in ['협력', '배신']:
                raise ValueError("반환값은 '협력' 또는 '배신'이어야 합니다.")
        except Exception as e:
            err_func, err_msg = 'func2', f"[vs {s1_name}] {e}"
            break
        # Score & History
        s1, s2 = PAYOFFS[(move1, move2)]
        score1 += s1; score2 += s2
        history1.append(move1); history2.append(move2)

    if err_func:
        return s1_id, s2_id, 0, 0, 0, err_func, f"{err_msg}\n{traceback.format_exc()}"
    else:
        return s1_id, s2_id, score1, score2, matchnum, None, None

# 정밀 검증(데몬 러너 기반, ns 타이머)
async def play_match_async_check(
    s1_id: int, s1_name: str, r1: PlayerRunner,
    s2_id: int, s2_name: str, r2: PlayerRunner
) -> Tuple:
    loop = asyncio.get_running_loop()
    history1, history2 = [], []
    remaining1_ns = MATCH_TIME_BUDGET_NS
    remaining2_ns = MATCH_TIME_BUDGET_NS

    for _ in range(110):
        # P1
        try:
            cap1_ns = remaining1_ns if remaining1_ns < PER_CALL_HARD_CAP_NS else PER_CALL_HARD_CAP_NS
            if cap1_ns <= 0:
                raise TimeoutError(f"총 시간 예산 초과 ({MATCH_TIME_BUDGET_SEC:.6f}s)")
            t0 = time.perf_counter_ns()
            move1 = await loop.run_in_executor(None, r1.move, history2, history1, cap1_ns / 1e9)
            remaining1_ns -= (time.perf_counter_ns() - t0)
            if move1 not in ['협력', '배신']:
                raise ValueError("반환값은 '협력' 또는 '배신'이어야 합니다.")
        except Exception as e:
            return (s1_id, s2_id, 0, 0, 0, 'func1', f"[vs {s2_name}] {e}\n{traceback.format_exc()}")

        # P2
        try:
            cap2_ns = remaining2_ns if remaining2_ns < PER_CALL_HARD_CAP_NS else PER_CALL_HARD_CAP_NS
            if cap2_ns <= 0:
                raise TimeoutError(f"총 시간 예산 초과 ({MATCH_TIME_BUDGET_SEC:.6f}s)")
            t0 = time.perf_counter_ns()
            move2 = await loop.run_in_executor(None, r2.move, history1, history2, cap2_ns / 1e9)
            remaining2_ns -= (time.perf_counter_ns() - t0)
            if move2 not in ['협력', '배신']:
                raise ValueError("반환값은 '협력' 또는 '배신'이어야 합니다.")
        except Exception as e:
            return (s1_id, s2_id, 0, 0, 0, 'func2', f"[vs {s1_name}] {e}\n{traceback.format_exc()}")

        history1.append(move1); history2.append(move2)

    return (s1_id, s2_id, 0, 0, 0, None, None)

def load_and_sanity_check_worker(args: Tuple) -> Tuple:
    """(id, code) -> (id, error_message | None)"""
    s_id, s_code = args
    try:
        # load_strategy_from_string는 샌드박싱된 exec를 사용
        func, err_msg = load_strategy_from_string(s_code)
        if err_msg:
            return s_id, f"[Compile Error] {err_msg}"
        # 1회 실행
        func(tuple(), tuple()) 
        return s_id, None
    except Exception as e:
        # 이 프로세스 내에서 발생하는 모든 오류 (e.g. while True의 Timeout)
        return s_id, f"[Sanity Check Error] {e}\n{traceback.format_exc()}"

def _run_pool_tasks(context, worker_func, tasks, timeout_per_task) -> List:
    results = []
    try:
        with context.Pool(processes=mp.cpu_count()) as pool:
            async_results = [
                pool.apply_async(worker_func, args=(args,))
                for args in tasks
            ]
            for i, res in enumerate(async_results):
                task_input = tasks[i]
                try:
                    result = res.get(timeout=timeout_per_task)
                    results.append(result)
                except mp.TimeoutError:
                    # ✅ (수정) v11과 달리, 타임아웃 시 에러 튜플을 반환
                    if worker_func == load_and_sanity_check_worker:
                        results.append((task_input[0], "[Sanity Check Error] 하드 타임아웃 (e.g., 'while True')"))
                    else:
                        results.append(None) 
                except Exception as e:
                    # ✅ (수정) v11과 달리, 오류 시 에러 튜플을 반환
                    if worker_func == load_and_sanity_check_worker:
                         results.append((task_input[0], f"[Sanity Check Error] Pool Worker Error: {e}"))
                    else:
                        results.append(None)
            return results
    except Exception as e:
        print(f"[치명적 오류] Pool 생성/관리 실패: {e}")
        return []


async def run_tournament(db: AsyncSession):
    """
    SANITY는 submit 시점에만 수행하여 _sanity_cache에 저장.
    토너먼트 시에는 캐시를 재사용(없으면 통과로 간주).
    Eager warmup: 통과한 전략에 대해 데몬 러너를 미리 준비.
    그 후 고속 풀리그 + RECHECK(데몬 러너로 정밀검증).
    """
    print("--- 🏁 토너먼트 시작 (sanity cached + daemon runners + eager warmup) ---")

    context = mp.get_context("spawn")
    strategy_map: Dict[int, Strategy] = {}
    total_scores: Dict[int, int] = {}
    count_rounds: Dict[int, int] = {}
    disqualified_players: Set[int] = set()

    # 전략 로드
    result = await db.execute(select(Strategy))
    strategies = result.scalars().all()
    if not strategies:
        print("--- ⚠️ 참가자가 없어 토너먼트 종료 ---")
        return

    for s in strategies:
        # 토너먼트 시작 시 runtime 관련 플래그는 초기화 (sanity 결과는 별도 캐시에서 관리)
        s.total_score = 0; s.avg_score = 0
        s.error_flag = False; s.error_message = None
        strategy_map[s.id] = s
        total_scores[s.id] = 0
        count_rounds[s.id] = 0

    # 0) SANITY 캐시 기반 필터링 + eager warmup
    print(f"--- 🔥 Eager warmup + sanity 캐시 확인 ---")
    passed: List[Tuple[int, str, str, Strategy]] = []
    for s in strategies:
        h = _code_hash(s.code_string)
        ok_msg = _sanity_cache.get((s.id, h))
        # 캐시에 있고 실패한 경우만 제외
        if ok_msg is not None and ok_msg[0] is False:
            disqualified_players.add(s.id)
            s.error_flag = True
            s.error_message = ok_msg[1] or "[Sanity Failed (cached)]"
            print(f"[SANITY 제외] {s.user_name}: {s.error_message.splitlines()[0]}")
            continue

        # 통과로 간주 → 러너 준비 시도
        passed.append((s.id, s.code_string, s.user_name, s))

    # Eager warmup: 통과 대상 러너를 미리 띄워 ready까지 완료
    loop = asyncio.get_running_loop()
    for s_id, code, s_name, _obj in passed:
        if s_id in disqualified_players:
            continue
        try:
            await loop.run_in_executor(None, get_or_create_runner, s_id, code)
        except Exception as e:
            # 러너 시작 자체 실패는 해당 전략만 제외
            if s_id not in disqualified_players:
                disqualified_players.add(s_id)
                strategy_map[s_id].error_flag = True
                strategy_map[s_id].error_message = f"[Runner Start Failed] {e}"

    # 1) 고속 풀리그
    print(f"--- ⚔️ 1단계 고속 풀리그 시작 (참가자: {len(passed)}명) ---")
    N = len(passed)
    async_results_with_info: List[Tuple] = []
    normal_results: List[Tuple] = []

    try:
        with context.Pool(processes=mp.cpu_count()) as pool:
            match_timeout = (MATCH_TIME_BUDGET_SEC * 2) + 5.0
            for i in range(N):
                for j in range(i + 1, N):
                    s1_tuple = passed[i]
                    s2_tuple = passed[j]
                    if s1_tuple[0] in disqualified_players or s2_tuple[0] in disqualified_players:
                        continue
                    args = (s1_tuple[0], s1_tuple[1], s1_tuple[2],
                            s2_tuple[0], s2_tuple[1], s2_tuple[2])
                    res = pool.apply_async(play_match_worker_no_signal, args=(args,))
                    async_results_with_info.append((res, s1_tuple, s2_tuple))

            if not async_results_with_info:
                print("--- ⚠️ 매치할 상대가 없어 토너먼트 종료 ---")
                await db.commit()
                return

            print(f"--- 🏃 {len(async_results_with_info)}개 매치 병렬 실행 ---")

            for res, s1_tuple, s2_tuple in async_results_with_info:
                s1_id, s1_code, s1_name, _ = s1_tuple
                s2_id, s2_code, s2_name, _ = s2_tuple
                if s1_id in disqualified_players or s2_id in disqualified_players:
                    continue

                try:
                    (s1_res_id, s2_res_id, sc1, sc2, rounds, err_func, err_msg) = \
                        await loop.run_in_executor(None, res.get, match_timeout)
                except mp.TimeoutError:
                    err_func = 'RECHECK'
                    err_msg = "매치 시간 초과 (Hard Timeout)"
                except Exception as e:
                    err_func = 'RECHECK'
                    err_msg = f"Pool Worker Error: {e}"

                if err_func == 'RECHECK':
                    print(f"--- 🛡️ {s1_name} vs {s2_name}: 정밀검증(데몬 러너) ---")
                    try:
                        r1 = _runner_registry.get(s1_id)
                        r2 = _runner_registry.get(s2_id)
                        if (r1 is None) or (not r1.p.is_alive()):
                            r1 = get_or_create_runner(s1_id, s1_code)
                        if (r2 is None) or (not r2.p.is_alive()):
                            r2 = get_or_create_runner(s2_id, s2_code)

                        (_, _, _, _, _, slow_err_func, slow_err_msg) = \
                            await play_match_async_check(s1_id, s1_name, r1, s2_id, s2_name, r2)

                        if slow_err_func:
                            culprit_id = s1_id if slow_err_func == 'func1' else s2_id
                            culprit_name = s1_name if slow_err_func == 'func1' else s2_name
                            print(f"--- ❌ 정밀검증 탈락: {culprit_name} ---")
                            if culprit_id not in disqualified_players:
                                disqualified_players.add(culprit_id)
                                strategy_map[culprit_id].error_flag = True
                                strategy_map[culprit_id].error_message = f"[정밀검증 런타임 오류] {slow_err_msg}"
                    except Exception as e:
                        print(f"[정밀검증 프레임워크 오류] {s1_name} vs {s2_name}: {e}")
                        continue

                elif err_func:
                    culprit_id = s1_id if err_func == 'func1' else s2_id
                    culprit_name = s1_name if err_func == 'func1' else s2_name
                    print(f"--- ❌ 런타임 탈락: {culprit_name} ---")
                    if culprit_id not in disqualified_players:
                        disqualified_players.add(culprit_id)
                        strategy_map[culprit_id].error_flag = True
                        strategy_map[culprit_id].error_message = f"[런타임 오류] {err_msg}"
                else:
                    normal_results.append((s1_id, s2_id, sc1, sc2, rounds))
    finally:
        # 데몬 러너는 재사용을 위해 유지
        pass

    # 2) 최종 집계 (탈락자 제외)
    for s1_id, s2_id, sc1, sc2, rounds in normal_results:
        if s1_id in disqualified_players or s2_id in disqualified_players:
            continue
        total_scores[s1_id] += sc1
        total_scores[s2_id] += sc2
        count_rounds[s1_id] += rounds
        count_rounds[s2_id] += rounds

    # 3) DB 업데이트
    for s in strategies:
        if s.error_flag or s.id in disqualified_players:
            s.avg_score = 0
            s.total_score = 0
            continue

        rounds = count_rounds[s.id]
        if rounds > 0:
            s.avg_score = int((total_scores[s.id] * (10**8)) / rounds + 0.5)
        else:
            s.avg_score = 0
        s.total_score = total_scores[s.id]

    await db.commit()
    print("--- ✅ 토너먼트 완료 ---")


# =========================
# 8) API 엔드포인트 (submit에서만 SANITY 수행)
# =========================
@app.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_strategy(submission: Submission, db: AsyncSession = Depends(get_db)):
    # 1) 사용자 전략 upsert (v11과 동일)
    result = await db.execute(
        select(Strategy).where(Strategy.user_name == submission.user_name)
    )
    db_strategy = result.scalar_one_or_none()
    if db_strategy:
        if not verify_password(submission.password, db_strategy.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="비밀번호가 틀렸습니다.")
        db_strategy.code_string = submission.code
    else:
        hashed_password = get_password_hash(submission.password)
        db_strategy = Strategy(
            user_name=submission.user_name,
            hashed_password=hashed_password,
            code_string=submission.code
        )
        db.add(db_strategy)
    
    await db.commit()
    await db.refresh(db_strategy)

    # 2) ✅ (수정) Sanity Check를 데몬 러너 생성으로 대체
    loop = asyncio.get_running_loop()
    h = _code_hash(db_strategy.code_string)
    ok = True
    msg = None

    try:
        # get_or_create_runner (v10.3/v11) 자체가 Sanity Check입니다.
        # 'while True' 코드는 'ready' 신호를 못 보내고 __init__에서 타임아웃됩니다.
        await loop.run_in_executor(
            None, 
            get_or_create_runner, # ❗️ (참고) v11 코드 상단에 이 함수가 정의되어 있어야 함
            db_strategy.id, 
            db_strategy.code_string
        )
        
    except Exception as e:
        # 러너 시작 실패 (e.g., 'while True', 컴파일 오류 등)
        ok = False
        msg = f"[Sanity Check Error] {e}" # e.g., "init timeout/error"

    # Sanity Check 결과를 캐시에 저장
    _sanity_cache[(db_strategy.id, h)] = (ok, msg)

    if not ok:
        db_strategy.error_flag = True
        db_strategy.error_message = msg
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"코드 오류(SANITY): {msg}"
        )
    else:
        db_strategy.error_flag = False
        db_strategy.error_message = None
        await db.commit()

    # 3) 토너먼트 실행 (v11과 동일)
    async with tournament_lock:
        await run_tournament(db)

    # 4) 토너먼트 결과 반영 확인 (v11과 동일)
    await db.refresh(db_strategy)
    if db_strategy.error_flag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"토너먼트 탈락: {db_strategy.error_message or '런타임 오류 또는 타임아웃'}"
        )
    
    # 5) 정상 통과 (v11과 동일)
    return {"detail": "코드가 성공적으로 제출/검증되었고 토너먼트에서도 통과했습니다."}

@app.get("/scoreboard", response_model=List[ScoreboardEntry])
async def get_scoreboard(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Strategy)
        .where(Strategy.error_flag == False)
        .order_by(Strategy.avg_score.desc())
    )
    strategies = result.scalars().all()
    return strategies

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(payload: Submission, db: AsyncSession = Depends(get_db)):
    """
    신규 회원가입: user_name 중복 불가.
    code는 비워두고 비번만 저장.
    """
    result = await db.execute(
        select(Strategy).where(Strategy.user_name == payload.user_name)
    )
    exists = result.scalar_one_or_none()
    if exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 존재하는 닉네임입니다."
        )
    hashed_password = get_password_hash(payload.password)
    user = Strategy(
        user_name=payload.user_name,
        hashed_password=hashed_password,
        code_string=""  # 초기엔 빈 코드
    )
    db.add(user)
    await db.commit()
    return {"detail": "회원가입이 완료되었습니다. 로그인해 주세요."}


class CodeRequest(BaseModel):
    user_name: str
    password: str

class CodeResponse(BaseModel):
    code: str

@app.post("/mycode", response_model=CodeResponse)
async def get_my_code(payload: CodeRequest, db: AsyncSession = Depends(get_db)):
    """
    로그인 정보로 본인 코드 문자열을 반환.
    """
    result = await db.execute(
        select(Strategy).where(Strategy.user_name == payload.user_name)
    )
    s = result.scalar_one_or_none()
    if not s or not verify_password(payload.password, s.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="닉네임 또는 비밀번호가 올바르지 않습니다."
        )
    return CodeResponse(code=s.code_string or "")

TOURNAMENT_INTERVAL_SEC = 300
async def tournament_scheduler():
    # 앱 기동 직후 약간 기다렸다가 시작 (DB 준비 등)
    await asyncio.sleep(3)
    while True:
        try:
            # 상태 플래그 (선택)
            app.state.tournament_running = True
            app.state.tournament_started_at = time.time()

            async with AsyncSessionLocal() as session:
                async with tournament_lock:
                    await run_tournament(session)

            app.state.tournament_last_finished_at = time.time()
        except Exception as e:
            print(f"[Scheduler] tournament run error: {e}")
        finally:
            app.state.tournament_running = False

        # 다음 주기까지 대기
        await asyncio.sleep(TOURNAMENT_INTERVAL_SEC)

@app.on_event("shutdown")
async def shutdown():
    try:
        app.state.tournament_task.cancel()
    except Exception:
        pass
    terminate_all_runners()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# 루트로 들어오면 로그인 페이지로 리다이렉트 (또는 직접 파일 반환)
@app.get("/", include_in_schema=False)
async def root():   
    return FileResponse("public/about.html")

app.mount("/", StaticFiles(directory="public", html=True), name="public")

# =========================
# 9) (선택) uvicorn으로 서버 실행 (테스트용)
# =========================
if __name__ == "__main__":
    import uvicorn
    # ✅ (변경) "127.0.0.1" -> "0.0.0.0"
    print("--- 백엔드 서버를 http://0.0.0.0:8000 에서 시작합니다 ---")
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    finally:
        terminate_all_runners()
