import time
import threading

class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        """
            因为threading类没有返回值,因此在此处重新定义MyThread类,使线程拥有返回值
        """
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # 接受返回值
        self.result = self.func(*self.args)

    def get_result(self):
        # 线程不结束,返回值为None
        try:
            return self.result
        except Exception:
            return None

def limit_decor(timeout, granularity):
    """
        timeout 最大允许执行时长, 单位:秒
        granularity 轮询间隔，间隔越短结果越精确同时cpu负载越高
        return 未超时返回被装饰函数返回值,超时则返回 None
    """
    def functions(func):
        def run(*args):
            thre_func = MyThread(target=func, args=args)
            thre_func.setDaemon(True)
            thre_func.start()
            sleep_num = int(timeout//granularity)
            for i in range(0, sleep_num):
                infor = thre_func.get_result()
                if infor:
                    return infor
                else:
                    time.sleep(granularity)
            print("#################gpt3 timeout#################")
            return None
        return run
    return functions
if __name__ == "__main__":   
    @limit_decor(1, 0.02)
    def test_func():
        time.sleep(3) 
        return 1

    print(test_func())
